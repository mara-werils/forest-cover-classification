"""Covariate and label shift between the four wilderness areas.

The group validation (src/robustness.py) shows that a model trained on
three areas reaches only 0.24-0.45 F1-macro on the fourth. This module
measures *why*, directly on the data: for every area it compares the
distribution of each feature and of the class with the other three areas.

* PSI (population stability index) on 10 quantile bins of the reference
  (= the other three areas); the usual reading is < 0.10 stable,
  0.10-0.25 moderate shift, > 0.25 major shift;
* the two-sample Kolmogorov-Smirnov statistic (largest gap between the
  two CDFs, 0..1) as a bin-free cross-check;
* total variation distance (TVD) for categorical columns: Soil and the
  target itself (label shift), plus the classes that the area has but the
  training areas never show.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.eda import BLUE_RAMP, TEXT, _save, _style

SHIFT_CSV = config.REPORTS_DIR / "area_shift.csv"
SHIFT_JSON = config.REPORTS_DIR / "area_shift.json"
PSI_BINS = 10
EPS = 1e-4


def psi(reference, current, bins=PSI_BINS):
    """Population stability index of `current` against `reference`.

    Bin edges are quantiles of the reference; duplicated edges (integer
    data with ties) are merged, so a feature with few distinct values
    simply gets fewer bins.
    """
    reference = np.asarray(reference, dtype="float64")
    current = np.asarray(current, dtype="float64")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_psi(reference, current):
    """PSI over category shares (categories missing on one side get EPS)."""
    ref = pd.Series(reference).value_counts(normalize=True)
    cur = pd.Series(current).value_counts(normalize=True)
    index = ref.index.union(cur.index)
    ref = ref.reindex(index, fill_value=0).clip(lower=EPS).to_numpy()
    cur = cur.reindex(index, fill_value=0).clip(lower=EPS).to_numpy()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def total_variation(reference, current):
    """Half the L1 distance between two categorical distributions (0..1)."""
    ref = pd.Series(reference).value_counts(normalize=True)
    cur = pd.Series(current).value_counts(normalize=True)
    index = ref.index.union(cur.index)
    return float(0.5 * (ref.reindex(index, fill_value=0)
                        - cur.reindex(index, fill_value=0)).abs().sum())


def severity(value):
    """Conventional reading of a PSI value."""
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "major"


def area_shift_table(df):
    """One row per (held-out area, feature): PSI, KS and severity."""
    area = df[config.WILDERNESS].to_numpy()
    rows = []
    for area_id, area_name in config.WILDERNESS_AREAS.items():
        inside = area == area_id
        if not inside.any():
            continue
        for col in config.NUMERIC_COLS:
            ref, cur = df.loc[~inside, col].to_numpy(), df.loc[inside, col].to_numpy()
            value = psi(ref, cur)
            rows.append({"area": area_name, "feature": col, "kind": "numeric",
                         "psi": value, "ks": float(stats.ks_2samp(ref, cur).statistic),
                         "tvd": np.nan, "severity": severity(value)})
        for col, label in ((config.SOIL, config.SOIL), (config.TARGET, "Cover_Type (label)")):
            ref, cur = df.loc[~inside, col].to_numpy(), df.loc[inside, col].to_numpy()
            value = categorical_psi(ref, cur)
            rows.append({"area": area_name, "feature": label, "kind": "categorical",
                         "psi": value, "ks": np.nan, "tvd": total_variation(ref, cur),
                         "severity": severity(value)})
    return pd.DataFrame(rows)


def area_summary(df, table):
    """Per area: size, mean numeric PSI, label shift and unseen classes."""
    area = df[config.WILDERNESS].to_numpy()
    target = df[config.TARGET].to_numpy()
    summary = []
    for area_id, area_name in config.WILDERNESS_AREAS.items():
        inside = area == area_id
        if not inside.any():
            continue
        part = table[table["area"] == area_name]
        numeric = part[part["kind"] == "numeric"]
        label = part[part["feature"] == "Cover_Type (label)"].iloc[0]
        soil = part[part["feature"] == config.SOIL].iloc[0]
        own, rest = set(np.unique(target[inside])), set(np.unique(target[~inside]))
        summary.append({
            "area": area_name,
            "cells": int(inside.sum()),
            "mean_numeric_psi": round(float(numeric["psi"].mean()), 4),
            "features_with_major_shift": int((numeric["severity"] == "major").sum()),
            "most_shifted_feature": numeric.loc[numeric["psi"].idxmax(), "feature"],
            "soil_tvd": round(float(soil["tvd"]), 4),
            "label_tvd": round(float(label["tvd"]), 4),
            "classes_only_in_this_area": sorted(config.COVER_TYPES[int(c)] for c in own - rest),
            "classes_missing_here": sorted(config.COVER_TYPES[int(c)] for c in rest - own),
        })
    return pd.DataFrame(summary)


def attach_group_validation(summary, path=None):
    """Add held-out F1-macro of LightGBM from reports/group_validation.json, if present."""
    path = path or config.REPORTS_DIR / "group_validation.json"
    if not path.exists():
        return summary
    records = pd.DataFrame(json.loads(path.read_text()))
    held_out = records[records["model"] == "lightgbm"].set_index("held_out_area")["f1_macro"]
    return summary.assign(heldout_f1_macro=summary["area"].map(held_out))


def plot_shift(table, summary):
    """Heat map of PSI (area x feature) and the label shift per area."""
    _style()
    pivot = table.pivot(index="area", columns="feature", values="psi")
    order = [c for c in config.NUMERIC_COLS + [config.SOIL, "Cover_Type (label)"]
             if c in pivot.columns]
    pivot = pivot.reindex(index=list(config.WILDERNESS_AREAS.values()), columns=order).dropna(
        how="all")
    labels = [c.replace("Horizontal_Distance_To_", "HD ").replace("Vertical_Distance_To_", "VD ")
              .replace("_", " ") for c in pivot.columns]

    fig, axes = plt.subplots(1, 2, figsize=(17, 4.6), gridspec_kw={"width_ratios": [3.2, 1]})
    ax = axes[0]
    image = ax.imshow(pivot.clip(upper=2.0), cmap=BLUE_RAMP, vmin=0, vmax=2.0, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot)), pivot.index, fontsize=9)
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 1.1 else TEXT)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01,
                 label="PSI vs other three areas (capped at 2)")
    ax.set_title("Covariate shift: each area against the other three "
                 "(PSI > 0.25 = major shift)")

    ax = axes[1]
    y = np.arange(len(summary))
    ax.barh(y - 0.18, summary["label_tvd"], height=0.34, color="#1c5cab",
            label="label shift (TVD of classes)")
    ax.barh(y + 0.18, summary["soil_tvd"], height=0.34, color="#86b6ef",
            label="soil shift (TVD of soils)")
    if "heldout_f1_macro" in summary:
        for i, f1 in enumerate(summary["heldout_f1_macro"]):
            if pd.notna(f1):
                ax.text(1.02, i, f"held-out F1 {f1:.2f}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y, summary["area"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.set_xlabel("Total variation distance (0 = same, 1 = disjoint)")
    ax.set_title("Label and soil shift")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1)
    fig.tight_layout()
    return _save(fig, "20_area_shift.png")


def run_shift(df, save=True):
    """Full analysis; stores CSV (long table), JSON (summary) and the figure."""
    table = area_shift_table(df)
    summary = attach_group_validation(area_summary(df, table))
    if save:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(config.REPORTS_DIR / SHIFT_CSV.name, index=False)
        payload = {
            "reference": "the other three wilderness areas",
            "psi_bins": PSI_BINS,
            "psi_reading": {"stable": "< 0.10", "moderate": "0.10-0.25", "major": "> 0.25"},
            "areas": summary.to_dict(orient="records"),
        }
        (config.REPORTS_DIR / SHIFT_JSON.name).write_text(
            json.dumps(payload, indent=2, default=str))
        plot_shift(table, summary)
    return table, summary


def print_shift(summary):
    """Console view used by the menu."""
    columns = ["area", "cells", "mean_numeric_psi", "features_with_major_shift",
               "most_shifted_feature", "soil_tvd", "label_tvd"]
    if "heldout_f1_macro" in summary:
        columns.append("heldout_f1_macro")
    print(summary[columns].to_string(index=False))
    for row in summary.itertuples():
        if row.classes_only_in_this_area:
            print(f"  only in {row.area}: {', '.join(row.classes_only_in_this_area)}")
