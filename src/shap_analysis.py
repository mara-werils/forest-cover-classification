"""Why does the model decide what it decides? Exact SHAP values from LightGBM.

LightGBM computes TreeSHAP itself (``predict(..., pred_contrib=True)``), so no
extra library is needed. For every cell and every class k the raw score
(log-odds before the softmax) is split exactly into a bias plus one
contribution per input column:

    f_k(x) = bias_k + sum_j phi_kj(x)

Three questions are answered on stratified subsamples of the test split
(TreeSHAP over 500 x 7 trees costs about 0.1-0.2 s per cell on a laptop CPU):

1. **Which inputs matter for which class?** Mean |phi_kj| over cells of
   class k, with the same number of cells per class so the rare classes are
   visible. The engineered columns are also summed into the groups of the
   feature ablation (``src/ablation.py``).
2. **Why are Spruce/Fir and Lodgepole Pine confused?** (69 % of all test
   errors.) For a cell the decision between the two is the margin
   f_Spruce - f_Lodgepole, whose contributions are
   phi_Spruce,j - phi_Lodgepole,j. Comparing these on correct cells and on
   errors shows which inputs pull the wrong way.
3. **Explaining one prediction**: the top contributions for a single record,
   e.g. the values entered in the prediction form.
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config, data_loader, features, models
from src.ablation import FEATURE_GROUPS
from src.eda import GRID, MUTED, TEXT, _save, _style
from src.evaluate import LABELS, SHORT

SPRUCE, LODGEPOLE = 1, 2
PER_CLASS = 200
PER_PAIR_GROUP = 150
FIGURE_NAME = "26_shap.png"
PAIR_GROUPS = {
    "Spruce/Fir, correct": (SPRUCE, SPRUCE),
    "Lodgepole, correct": (LODGEPOLE, LODGEPOLE),
    "Spruce/Fir predicted as Lodgepole": (SPRUCE, LODGEPOLE),
    "Lodgepole predicted as Spruce/Fir": (LODGEPOLE, SPRUCE),
}


def _path(name):
    """Output paths are built at call time, so tests can redirect REPORTS_DIR."""
    return config.REPORTS_DIR / name


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------
def split_pipeline(pipeline):
    """(preprocessing, LightGBM estimator, column names the estimator sees)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "booster_"):
        raise ValueError("SHAP via pred_contrib needs a fitted LightGBM model "
                         f"(got {type(model).__name__}).")
    return pipeline.named_steps["prep"], model, features.feature_names_after_preprocessing("tree")


def shap_values(pipeline, X):
    """SHAP values as an array (cells, classes, columns) plus the bias (cells, classes).

    Classes follow ``model.classes_``; the sum over columns plus the bias equals
    the raw score exactly.
    """
    prep, model, names = split_pipeline(pipeline)
    transformed = prep.transform(X)
    contrib = model.predict(transformed, pred_contrib=True)
    n_classes = len(model.classes_)
    cube = np.asarray(contrib).reshape(len(X), n_classes, len(names) + 1)
    return cube[:, :, :-1], cube[:, :, -1]


def class_position(model, label):
    return int(np.flatnonzero(np.asarray(model.classes_) == label)[0])


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
def balanced_sample(y, per_class=PER_CLASS, seed=config.RANDOM_STATE):
    """Index labels with up to ``per_class`` cells of every class."""
    rng = np.random.default_rng(seed)
    picked = []
    for label in LABELS:
        index = y.index[y == label]
        picked.extend(rng.choice(index, size=min(per_class, len(index)), replace=False))
    return pd.Index(picked)


def pair_sample(y, predictions, per_group=PER_PAIR_GROUP, seed=config.RANDOM_STATE):
    """Up to ``per_group`` cells for each (true, predicted) group of PAIR_GROUPS."""
    rng = np.random.default_rng(seed)
    groups = {}
    for name, (true, pred) in PAIR_GROUPS.items():
        index = y.index[(y == true) & (predictions == pred)]
        groups[name] = pd.Index(rng.choice(index, size=min(per_group, len(index)),
                                           replace=False))
    return groups


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def per_class_importance(values, y, classes, names):
    """Mean |phi| per (column, class) over cells of that class; plus an overall column."""
    y = np.asarray(y)
    table = pd.DataFrame(index=names)
    for position, label in enumerate(classes):
        mask = y == label
        if mask.any():
            table[SHORT[int(label)]] = np.abs(values[mask, position, :]).mean(axis=0)
    table["all classes"] = table.mean(axis=1)
    return table.sort_values("all classes", ascending=False)


def group_importance(values, names):
    """Mean |sum of phi over a group| per class, for raw columns and engineered groups.

    Summing before the absolute value keeps effects of the same group that cancel
    out from looking important.
    """
    position = {name: i for i, name in enumerate(names)}
    engineered = {c for cols in FEATURE_GROUPS.values() for c in cols}
    groups = {name: [name] for name in names if name not in engineered}
    groups.update({f"[{g}]": cols for g, cols in FEATURE_GROUPS.items()})
    rows = {}
    for group, columns in groups.items():
        summed = values[:, :, [position[c] for c in columns]].sum(axis=2)
        rows[group] = np.abs(summed).mean()
    series = pd.Series(rows).sort_values(ascending=False)
    return (series / series.sum()).rename("share_of_attribution")


def pair_margin_contributions(values, classes, model_classes=None):
    """phi_Spruce - phi_Lodgepole per cell and column (positive = towards Spruce/Fir)."""
    classes = list(model_classes if model_classes is not None else classes)
    return values[:, classes.index(SPRUCE), :] - values[:, classes.index(LODGEPOLE), :]


def pair_table(margins, groups, names):
    """Mean pair-margin contribution of every column in every group of cells."""
    table = pd.DataFrame(index=names)
    for name, rows in groups.items():
        if len(rows):
            table[name] = margins[rows].mean(axis=0)
    return table


def crossover_elevation(elevation, contribution):
    """Elevation where the Elevation contribution to the pair margin changes sign.

    Uses the median contribution in 50 m bands and linear interpolation at
    the first sign change from Lodgepole (low) to Spruce/Fir (high).
    """
    frame = pd.DataFrame({"band": (np.asarray(elevation) // 50) * 50 + 25,
                          "phi": np.asarray(contribution)})
    medians = frame.groupby("band")["phi"].median()
    medians = medians[frame.groupby("band").size() >= 5]
    signs = np.sign(medians.to_numpy())
    for i in range(len(signs) - 1):
        if signs[i] < 0 <= signs[i + 1]:
            x0, x1 = medians.index[i], medians.index[i + 1]
            y0, y1 = medians.iloc[i], medians.iloc[i + 1]
            return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    return None


def explain_record(record, pipeline=None, top=5):
    """Top contributions to the predicted class for one record (dict of input values)."""
    pipeline = pipeline or models.load_model()
    frame = features.to_input_frame(record)
    values, bias = shap_values(pipeline, frame)
    _, model, names = split_pipeline(pipeline)
    position = int(np.argmax(values[0].sum(axis=1) + bias[0]))
    label = int(model.classes_[position])
    transformed = pipeline.named_steps["prep"].transform(frame)[0]
    contributions = pd.DataFrame({"feature": names, "value": transformed,
                                  "contribution": values[0, position]})
    contributions = contributions.reindex(
        contributions["contribution"].abs().sort_values(ascending=False).index).head(top)
    return {"cover_type": config.COVER_TYPES[label], "bias": round(float(bias[0, position]), 3),
            "score": round(float(values[0, position].sum() + bias[0, position]), 3),
            "top_contributions": contributions.round(3).to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_shap(df=None, per_class=PER_CLASS, per_group=PER_PAIR_GROUP, save=True):
    """All three analyses for the saved model (about 5-10 minutes on 4 threads)."""
    df = df if df is not None else data_loader.load_parquet()
    pipeline = models.load_model()
    _, model, names = split_pipeline(pipeline)
    classes = [int(c) for c in model.classes_]
    _, X_test, _, y_test = features.split_data(df)
    predictions = pd.Series(pipeline.predict(X_test), index=X_test.index)

    balanced = balanced_sample(y_test, per_class)
    groups = pair_sample(y_test, predictions, per_group)
    union = balanced.union(pd.Index(np.concatenate([g.to_numpy() for g in groups.values()])))
    start = time.perf_counter()
    print(f"TreeSHAP for {len(union):,} test cells ...")
    values, _ = shap_values(pipeline, X_test.loc[union])
    print(f"done in {time.perf_counter() - start:.0f} s")
    row = {label: i for i, label in enumerate(union)}

    rows_balanced = [row[i] for i in balanced]
    importance = per_class_importance(values[rows_balanced], y_test.loc[balanced], classes, names)
    grouped = group_importance(values[rows_balanced], names)

    margins = pair_margin_contributions(values, classes)
    group_rows = {name: [row[i] for i in index] for name, index in groups.items()}
    pair = pair_table(margins, group_rows, names)
    elevation_pos = names.index("Elevation")
    pair_rows = sorted({r for rows in group_rows.values() for r in rows})
    pair_index = union[pair_rows]
    elevation = X_test.loc[pair_index, "Elevation"].to_numpy()
    crossover = crossover_elevation(elevation, margins[pair_rows, elevation_pos])
    errors = [row[i] for name, index in groups.items() if "predicted" in name for i in index]
    error_elevation = X_test.loc[union[errors], "Elevation"]

    permutation = None
    perm_path = config.REPORTS_DIR / "permutation_importance.csv"
    if perm_path.exists():
        perm = pd.read_csv(perm_path)
        perm = perm.set_index(perm.columns[0]).iloc[:, 0]
        common = [n for n in importance.index if n in perm.index]
        if len(common) >= 3:
            rho = stats.spearmanr(importance.loc[common, "all classes"], perm.loc[common])
            permutation = {"spearman_rho": round(float(rho.statistic), 3),
                           "features_compared": len(common)}

    example = explain_record(_example_record(), pipeline)
    summary = {
        "cells_explained": len(union), "per_class": per_class, "per_pair_group": per_group,
        "top_features_overall": importance["all classes"].head(5).round(4).to_dict(),
        "attribution_share_by_group": grouped.round(4).to_dict(),
        "spruce_lodgepole_crossover_elevation_m": round(crossover) if crossover else None,
        "pair_error_elevation_median_m": float(error_elevation.median()),
        "pair_error_elevation_iqr_m": [float(error_elevation.quantile(0.25)),
                                       float(error_elevation.quantile(0.75))],
        "agreement_with_permutation_importance": permutation,
        "example_record": example,
    }
    print(importance.round(3).to_string())
    print(grouped.round(3).to_string())
    print(pair.round(3).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "example_record"}, indent=2))
    print(json.dumps(example, indent=2))

    if save:
        importance.round(5).to_csv(_path("shap_per_class.csv"), index_label="feature")
        grouped.round(5).to_csv(_path("shap_groups.csv"), index_label="group")
        pair.round(5).to_csv(_path("shap_spruce_vs_lodgepole.csv"), index_label="feature")
        _path("shap_summary.json").write_text(json.dumps(summary, indent=2))
        plot_shap(importance, pair, elevation, margins[pair_rows, elevation_pos],
                  y_test.loc[pair_index].to_numpy(), crossover)
    return importance, pair, summary


def _example_record():
    from src import predict

    return predict.EXAMPLE_RECORD


def plot_shap(importance, pair, elevation, elevation_phi, true_labels, crossover, top=10):
    """Per-class heat map, Spruce/Fir vs Lodgepole contributions, Elevation dependence."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # 1. Mean |SHAP| per class.
    ax = axes[0]
    heat = importance.drop(columns="all classes").head(top)
    image = ax.imshow(heat.to_numpy(), cmap="Greens", aspect="auto")
    ax.set_xticks(range(heat.shape[1]), heat.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(heat)), heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if value > heat.to_numpy().max() * 0.6 else TEXT)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="mean |SHAP| (log-odds)")
    ax.set_title(f"Which inputs drive each class (top {top}, cells of that class)",
                 fontsize=9, fontweight="bold")
    ax.grid(False)

    # 2. Pair margin: errors vs correct cells.
    ax = axes[1]
    order = pair.abs().max(axis=1).sort_values(ascending=False).head(8).index[::-1]
    shown = pair.loc[order]
    y = np.arange(len(shown))
    palette = {"Spruce/Fir, correct": "#1F4D36", "Lodgepole, correct": "#8FB996",
               "Spruce/Fir predicted as Lodgepole": "#D08C2E",
               "Lodgepole predicted as Spruce/Fir": "#E8C07D"}
    width = 0.8 / shown.shape[1]
    for k, column in enumerate(shown.columns):
        ax.barh(y + (k - (shown.shape[1] - 1) / 2) * width, shown[column], width * 0.95,
                color=palette.get(column, MUTED), label=column)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, shown.index)
    ax.set_xlabel("mean contribution to f_Spruce - f_Lodgepole\n(> 0 pushes towards Spruce/Fir)")
    ax.set_title("Spruce/Fir vs Lodgepole Pine: what pulls the wrong way",
                 fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="y", visible=False)

    # 3. Elevation dependence for the pair.
    ax = axes[2]
    for label, color, name in ((SPRUCE, "#1F4D36", "true Spruce/Fir"),
                               (LODGEPOLE, "#D08C2E", "true Lodgepole")):
        mask = true_labels == label
        ax.scatter(np.asarray(elevation)[mask], np.asarray(elevation_phi)[mask], s=7, alpha=0.5,
                   color=color, label=name, edgecolors="none")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    if crossover:
        ax.axvline(crossover, color=TEXT, linewidth=0.8, linestyle="--")
        low = ax.get_ylim()[0]
        ax.text(crossover + 20, low + 0.08 * (ax.get_ylim()[1] - low),
                f"sign switch at ~{crossover:.0f} m", fontsize=7.5, color=TEXT)
    ax.set_xlabel("Elevation, m")
    ax.set_ylabel("Elevation contribution to f_Spruce - f_Lodgepole")
    ax.set_title("Elevation decides the pair only away from the switch",
                 fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left", markerscale=2)

    for ax in axes[1:]:
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].tick_params(labelsize=8)
    fig.tight_layout()
    return _save(fig, FIGURE_NAME)


if __name__ == "__main__":
    run_shap()
