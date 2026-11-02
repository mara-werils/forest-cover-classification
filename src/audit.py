"""Data quality audit: missing values, duplicates, one-hot integrity,
physical ranges, IQR outliers and class balance.

The result is a plain dict (saved as JSON) so it can be shown in the
console menu, in Streamlit and quoted in the report.
"""

import json

import numpy as np

from src import config


def missing_values(df):
    """Number of missing values per column (only non-zero ones)."""
    counts = df.isna().sum()
    return {col: int(n) for col, n in counts.items() if n > 0}


def duplicate_report(df):
    """Exact duplicates and feature duplicates with conflicting labels."""
    features = config.NUMERIC_COLS + config.WILDERNESS_COLS + config.SOIL_COLS
    full_dups = int(df.duplicated(subset=features + [config.TARGET]).sum())
    feature_dups = int(df.duplicated(subset=features).sum())
    # Same features, different cover type -> irreducible label noise.
    labels_per_group = df.groupby(features, observed=True)[config.TARGET].nunique()
    conflicting = int((labels_per_group > 1).sum())
    return {
        "exact_duplicate_rows": full_dups,
        "duplicate_feature_vectors": feature_dups,
        "feature_vectors_with_conflicting_labels": conflicting,
    }


def one_hot_integrity(df):
    """Every row must have exactly one wilderness area and one soil type."""
    result = {}
    for name, cols in (("wilderness", config.WILDERNESS_COLS), ("soil", config.SOIL_COLS)):
        active = df[cols].sum(axis=1)
        result[name] = {
            "rows_with_none": int((active == 0).sum()),
            "rows_with_several": int((active > 1).sum()),
            "never_used_columns": [c for c in cols if df[c].sum() == 0],
        }
    return result


def range_violations(df):
    """Values outside the physical range described in covtype.info."""
    violations = {}
    for col, (low, high) in config.VALID_RANGES.items():
        bad = np.zeros(len(df), dtype=bool)
        if low is not None:
            bad |= df[col].to_numpy() < low
        if high is not None:
            bad |= df[col].to_numpy() > high
        violations[col] = int(bad.sum())
    return violations


def iqr_outliers(df, columns=None, k=1.5):
    """Tukey IQR rule per numeric column: count, share and fences."""
    columns = columns or config.NUMERIC_COLS
    summary = {}
    for col in columns:
        values = df[col].astype("float32")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        count = int(((values < low) | (values > high)).sum())
        summary[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "lower_fence": float(low),
            "upper_fence": float(high),
            "outliers": count,
            "outliers_pct": round(100 * count / len(df), 2),
        }
    return summary


def class_balance(df):
    """Count and share of each cover type, plus the imbalance ratio."""
    counts = df[config.TARGET].value_counts().sort_index()
    shares = counts / counts.sum()
    return {
        "counts": {config.COVER_TYPES[int(k)]: int(v) for k, v in counts.items()},
        "shares_pct": {config.COVER_TYPES[int(k)]: round(100 * v, 2) for k, v in shares.items()},
        "imbalance_ratio_max_min": round(float(counts.max() / counts.min()), 1),
    }


def suspicious_values(df):
    """Values that are legal but physically doubtful."""
    return {
        # Hillshade 0 at 3pm means full shadow on a summer afternoon.
        "hillshade_3pm_zero": int((df["Hillshade_3pm"] == 0).sum()),
        "hillshade_9am_zero": int((df["Hillshade_9am"] == 0).sum()),
        # Aspect of exactly 360 is the same direction as 0.
        "aspect_equals_360": int((df["Aspect"] == 360).sum()),
        "negative_vertical_distance": int((df["Vertical_Distance_To_Hydrology"] < 0).sum()),
    }


def run_audit(df, save=True):
    """Run all checks and (optionally) store the report as JSON."""
    report = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing_values": missing_values(df),
        "duplicates": duplicate_report(df),
        "one_hot_integrity": one_hot_integrity(df),
        "range_violations": range_violations(df),
        "iqr_outliers": iqr_outliers(df),
        "suspicious_values": suspicious_values(df),
        "class_balance": class_balance(df),
        "describe": df[config.NUMERIC_COLS].describe().round(2).to_dict(),
    }
    if save:
        config.AUDIT_JSON.write_text(json.dumps(report, indent=2))
    return report


def print_audit(report):
    """Short human-readable version of the audit for the console."""
    print(f"Rows: {report['shape']['rows']:,}  Columns: {report['shape']['columns']}")
    print(f"Missing values: {report['missing_values'] or 'none'}")
    for key, value in report["duplicates"].items():
        print(f"{key}: {value:,}")
    for name, info in report["one_hot_integrity"].items():
        print(f"{name}: none={info['rows_with_none']}, several={info['rows_with_several']}")
    bad = {k: v for k, v in report["range_violations"].items() if v}
    print(f"Range violations: {bad or 'none'}")
    print("IQR outliers (% of rows):")
    for col, info in report["iqr_outliers"].items():
        print(f"  {col:<36} {info['outliers_pct']:>6.2f}%")
    print(f"Suspicious values: {report['suspicious_values']}")
    print("Class shares (%):")
    for name, share in report["class_balance"]["shares_pct"].items():
        print(f"  {name:<20} {share:>6.2f}")
