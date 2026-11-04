"""Every audit check against a frame where the right answer is known."""

import json

import numpy as np
import pytest

from src import audit, config


def test_clean_frame_has_no_findings(make_frame):
    df = make_frame()
    assert audit.missing_values(df) == {}
    assert sum(audit.range_violations(df).values()) == 0
    integrity = audit.one_hot_integrity(df)
    for block in ("wilderness", "soil"):
        assert integrity[block]["rows_with_none"] == 0
        assert integrity[block]["rows_with_several"] == 0


def test_missing_values_are_counted_per_column(make_frame):
    df = make_frame().astype({"Slope": "float32", "Elevation": "float32"})
    df.loc[[0, 5, 9], "Slope"] = np.nan
    df.loc[3, "Elevation"] = np.nan
    assert audit.missing_values(df) == {"Elevation": 1, "Slope": 3}


def test_exact_duplicates_and_conflicting_labels(make_frame):
    df = make_frame()
    # Row 1 becomes an exact copy of row 0.
    df.iloc[1] = df.iloc[0]
    # Row 3 gets the features of row 2 but keeps another label.
    features = config.NUMERIC_COLS + config.WILDERNESS_COLS + config.SOIL_COLS
    df.loc[3, features] = df.loc[2, features].to_numpy()
    df.loc[3, config.TARGET] = 7 if df.loc[2, config.TARGET] != 7 else 1

    report = audit.duplicate_report(df)
    assert report["exact_duplicate_rows"] == 1
    assert report["duplicate_feature_vectors"] == 2
    assert report["feature_vectors_with_conflicting_labels"] == 1


def test_one_hot_rows_with_none_or_several_flags(make_frame):
    df = make_frame()
    df.loc[0, config.WILDERNESS_COLS] = 0          # no area at all
    df.loc[1, config.SOIL_COLS[:3]] = 1            # three soils at once
    integrity = audit.one_hot_integrity(df)
    assert integrity["wilderness"]["rows_with_none"] == 1
    assert integrity["soil"]["rows_with_several"] >= 1


def test_never_used_one_hot_columns_are_listed(make_frame):
    df = make_frame(n=35)   # soils 36-40 never appear in 35 cycled rows
    unused = audit.one_hot_integrity(df)["soil"]["never_used_columns"]
    assert unused == [f"Soil_Type{i}" for i in range(36, 41)]


def test_range_violations_catch_both_bounds(make_frame):
    df = make_frame()
    df.loc[0, "Slope"] = 95                         # above 90
    df.loc[1, "Hillshade_Noon"] = 300               # above 255
    df.loc[2, "Horizontal_Distance_To_Roadways"] = -5   # below 0
    violations = audit.range_violations(df)
    assert violations["Slope"] == 1
    assert violations["Hillshade_Noon"] == 1
    assert violations["Horizontal_Distance_To_Roadways"] == 1
    assert sum(violations.values()) == 3


def test_iqr_fences_match_tukey_rule(make_frame):
    df = make_frame(n=100)
    df["Elevation"] = np.arange(100) + 2000         # q1=2024.75, q3=2074.25
    df.loc[99, "Elevation"] = 9000                  # one obvious outlier
    info = audit.iqr_outliers(df, columns=["Elevation"])["Elevation"]
    q1, q3 = df["Elevation"].astype("float32").quantile([0.25, 0.75])
    assert info["lower_fence"] == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert info["upper_fence"] == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert info["outliers"] == 1
    assert info["outliers_pct"] == 1.0


def test_wider_iqr_multiplier_flags_fewer_rows(make_frame):
    df = make_frame(n=700)
    tight = audit.iqr_outliers(df, k=0.5)
    loose = audit.iqr_outliers(df, k=3.0)
    for col in config.NUMERIC_COLS:
        assert loose[col]["outliers"] <= tight[col]["outliers"]


def test_class_balance_shares_and_ratio(make_frame):
    df = make_frame(n=70)
    df.loc[df[config.TARGET] == 4, config.TARGET] = 2   # class 4 disappears
    df.loc[0, config.TARGET] = 4                        # ...except one row
    balance = audit.class_balance(df)
    assert sum(balance["counts"].values()) == 70
    assert sum(balance["shares_pct"].values()) == pytest.approx(100, abs=0.05)
    assert balance["counts"]["Cottonwood/Willow"] == 1
    lodgepole = balance["counts"]["Lodgepole Pine"]
    assert balance["imbalance_ratio_max_min"] == pytest.approx(lodgepole / 1, rel=0.01)


def test_suspicious_values(make_frame):
    df = make_frame()
    df.loc[[0, 1], "Hillshade_3pm"] = 0
    df.loc[2, "Aspect"] = 360
    df["Vertical_Distance_To_Hydrology"] = 10
    df.loc[[3, 4, 5], "Vertical_Distance_To_Hydrology"] = -20
    found = audit.suspicious_values(df)
    assert found["hillshade_3pm_zero"] == 2
    assert found["aspect_equals_360"] == 1
    assert found["negative_vertical_distance"] == 3


def test_run_audit_writes_json(make_frame, tmp_reports):
    df = make_frame()
    report = audit.run_audit(df, save=True)
    stored = json.loads(config.AUDIT_JSON.read_text())
    assert stored["shape"] == {"rows": len(df), "columns": df.shape[1]}
    assert set(stored) == set(report)


def test_print_audit_is_readable(sample, capsys):
    audit.print_audit(audit.run_audit(sample, save=False))
    out = capsys.readouterr().out
    assert "Rows: 1,0" in out
    assert "Class shares" in out
    for name in config.COVER_TYPES.values():
        assert name in out


def test_class_extremes_are_not_outliers_inside_their_class(make_frame):
    """Two altitude belts: the high belt is a global outlier but normal for its class."""
    df = make_frame(n=700)
    rng = np.random.default_rng(1)
    low_belt = df[config.TARGET] != 7
    df.loc[low_belt, "Elevation"] = rng.integers(2400, 2600, low_belt.sum())
    df.loc[~low_belt, "Elevation"] = rng.integers(3600, 3700, (~low_belt).sum())
    df.loc[0, "Elevation"] = 5000        # row 0 is class 1: odd globally AND in its class

    info = audit.class_conditional_outliers(df, columns=["Elevation"])["Elevation"]
    krummholz_rows = int((~low_belt).sum())
    assert info["global_outliers"] == krummholz_rows + 1
    assert info["explained_by_class"] == krummholz_rows
    assert info["outlier_in_both"] == 1
    assert info["within_class_outliers"] == 1
    assert info["within_class_pct_by_class"]["Krummholz"] == 0


def test_run_audit_contains_the_class_conditional_section(sample):
    report = audit.run_audit(sample, save=False)
    by_class = report["iqr_outliers_by_class"]
    assert set(by_class) == set(config.NUMERIC_COLS)
    for col, info in by_class.items():
        assert info["global_outliers"] == report["iqr_outliers"][col]["outliers"]
        assert info["explained_by_class"] + info["outlier_in_both"] == info["global_outliers"]
