"""Feature ablation: do the engineered features help, and which ones?

``features.add_engineered_features`` adds 8 columns to the 10 raw numeric
inputs. This module measures their effect instead of assuming it:

* ``all_features``  - the configuration used by every model in step 5;
* ``raw_only``      - the 10 raw numeric columns + Wilderness + Soil;
* ``without_<g>``   - all features minus one group ``g`` (leave-one-group-out),
  which shows how much each group adds on top of the others.

Every model keeps its tuned hyper-parameters from ``reports/tuning_results.json``
and is trained on the same stratified 80/20 split as in step 5, so the
``all_features`` row reproduces the main experiment and the other rows differ
from it only in the input columns. The hyper-parameters were tuned with all
features, which can only favour that configuration: a gain of the engineered
features smaller than the noise is therefore a strong "they do not matter".
"""

import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src import config, data_loader, features, models
from src.eda import GRID, MUTED, TEXT, _save, _style
from src.evaluate import LABELS, SHORT

# The 8 engineered columns in 4 groups that are added or removed together.
FEATURE_GROUPS = {
    "hydrology_geometry": ["Euclidean_Distance_To_Hydrology", "Water_Elevation"],
    "distance_combinations": ["Hydro_Plus_Fire", "Abs_Hydro_Minus_Road",
                              "Abs_Fire_Minus_Road"],
    "aspect_sin_cos": ["Aspect_Sin", "Aspect_Cos"],
    "hillshade_mean": ["Hillshade_Mean"],
}
ABLATION_MODELS = ("logreg", "random_forest", "lightgbm")
REFERENCE_SET = "all_features"
FIGURE_NAME = "23_feature_ablation.png"
MODEL_TITLES = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
                "hist_gb": "HistGradientBoosting", "lightgbm": "LightGBM"}


def ablation_csv():
    """Output paths are built at call time, so tests can redirect REPORTS_DIR."""
    return config.REPORTS_DIR / "feature_ablation.csv"


def ablation_json():
    return config.REPORTS_DIR / "feature_ablation.json"


def feature_sets():
    """Numeric input columns of every compared configuration (dict, ordered)."""
    sets = {REFERENCE_SET: list(features.NUMERIC_MODEL_COLS)}
    for group, columns in FEATURE_GROUPS.items():
        sets[f"without_{group}"] = [c for c in features.NUMERIC_MODEL_COLS if c not in columns]
    sets["raw_only"] = list(config.NUMERIC_COLS)
    return sets


def build_ablation_pipeline(name, numeric_cols, params=None):
    """Same pipeline as ``models.build_pipeline``, restricted to ``numeric_cols``."""
    spec = models.model_specs()[name]
    if spec["kind"] == "linear":
        prep = features.linear_preprocessor(numeric_cols)
    else:
        prep = features.tree_preprocessor(numeric_cols)
    pipeline = Pipeline([("prep", prep), ("model", clone(spec["estimator"]))])
    if name == "hist_gb":
        # Wilderness and Soil stay the last two columns, whatever the numeric set.
        n = len(numeric_cols)
        pipeline.set_params(model__categorical_features=[n, n + 1])
    if params:
        pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})
    return pipeline


def fit_and_score(name, numeric_cols, params, X_train, y_train, X_test, y_test):
    """Train one configuration and return its quality and cost."""
    pipeline = build_ablation_pipeline(name, numeric_cols, params)
    with models.PeakMemory() as memory, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = pipeline.predict(X_test)
    predict_seconds = time.perf_counter() - start
    per_class = f1_score(y_test, predictions, labels=LABELS, average=None, zero_division=0)
    row = {
        "n_numeric_features": len(numeric_cols),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "f1_macro": round(f1_score(y_test, predictions, average="macro", zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, predictions), 4),
        "train_seconds": round(train_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
        "peak_memory_mb": round(memory.peak_mb, 1),
    }
    for label, value in zip(LABELS, per_class):
        row[f"f1_{SHORT[label]}"] = round(float(value), 4)
    return row


def add_deltas(table):
    """F1-macro and accuracy relative to the all-features row of the same model."""
    reference = table[table["feature_set"] == REFERENCE_SET].set_index("model")
    table = table.copy()
    table["delta_f1_macro"] = (table["f1_macro"]
                               - table["model"].map(reference["f1_macro"])).round(4)
    table["delta_accuracy"] = (table["accuracy"]
                               - table["model"].map(reference["accuracy"])).round(4)
    return table


def verdicts(table, tolerance=0.002):
    """Plain-language conclusions, one per model.

    ``tolerance`` is the F1-macro change treated as "no effect" (0.2 points
    by default, a deliberately small threshold).
    """
    lines = []
    for model, part in table.groupby("model", sort=False):
        part = part.set_index("feature_set")
        gain = part.loc[REFERENCE_SET, "f1_macro"] - part.loc["raw_only", "f1_macro"]
        drops = part.drop(index=[REFERENCE_SET, "raw_only"])["delta_f1_macro"]
        key_group = drops.idxmin().removeprefix("without_")
        if gain > tolerance:
            effect = f"the engineered features add {gain:+.4f} F1-macro over the raw inputs"
        elif gain < -tolerance:
            effect = f"the engineered features cost {gain:+.4f} F1-macro (raw inputs are better)"
        else:
            effect = f"the engineered features change F1-macro by {gain:+.4f}, i.e. not at all"
        if drops.min() < -tolerance:
            effect += (f"; removing '{key_group}' hurts most "
                       f"({drops.min():+.4f})")
        else:
            effect += "; no single group matters on its own (all drops within noise)"
        lines.append(f"{MODEL_TITLES.get(model, model)}: {effect}.")
    return lines


def run_ablation(df=None, model_names=ABLATION_MODELS, train_rows=None, save=True):
    """Train every model on every feature set and compare with the reference.

    ``train_rows`` (optional) trains on a stratified subsample of the training
    split for a quick run; the test split is always the full 20 %.
    About 20-30 minutes on 4 CPU threads with the full training split.
    """
    df = df if df is not None else data_loader.load_parquet()
    if not config.TUNING_JSON.exists():
        raise FileNotFoundError(f"{config.TUNING_JSON} not found. Train the models first "
                                "(menu step 5).")
    tuning = json.loads(config.TUNING_JSON.read_text())
    X_train, X_test, y_train, y_test = features.split_data(df)
    if train_rows:
        X_train, y_train = features.stratified_subsample(X_train, y_train, train_rows)
    sets = feature_sets()
    print(f"Feature ablation: {len(model_names)} models x {len(sets)} feature sets, "
          f"train {len(X_train):,} rows, test {len(X_test):,} rows")

    rows = []
    for name in model_names:
        params = tuning.get(name, {}).get("best_params", {})
        for set_name, columns in sets.items():
            row = {"model": name, "feature_set": set_name,
                   **fit_and_score(name, columns, params, X_train, y_train, X_test, y_test)}
            rows.append(row)
            print(f"[{name:13s}] {set_name:32s} F1-macro={row['f1_macro']:.4f} "
                  f"acc={row['accuracy']:.4f} train={row['train_seconds']:.0f}s")

    table = add_deltas(pd.DataFrame(rows))
    conclusions = verdicts(table)
    for line in conclusions:
        print(" * " + line)
    if save:
        table.to_csv(ablation_csv(), index=False)
        ablation_json().write_text(json.dumps({
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_groups": FEATURE_GROUPS,
            "hyper_parameters": "tuned with all features (reports/tuning_results.json)",
            "verdicts": conclusions,
            "results": table.to_dict(orient="records"),
        }, indent=2))
        plot_ablation(table)
    return table


def _set_label(set_name):
    if set_name == "raw_only":
        return "raw inputs only"
    return "without " + set_name.removeprefix("without_").replace("_", " ")


def plot_ablation(table):
    """Change in F1-macro against all features, one panel per model."""
    _style()
    model_names = list(dict.fromkeys(table["model"]))
    others = table[table["feature_set"] != REFERENCE_SET]
    # One x scale for all panels, so the bars of different models are comparable.
    span = max(others["delta_f1_macro"].abs().max(), 0.002)
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.6 * len(model_names), 3.8),
                             sharey=True, squeeze=False)
    for ax, model in zip(axes[0], model_names):
        part = others[others["model"] == model]
        deltas = part["delta_f1_macro"].to_numpy()
        y = np.arange(len(part))[::-1]
        colors = ["#2E7D4F" if d < 0 else "#D08C2E" for d in deltas]
        ax.set_axisbelow(True)
        ax.barh(y, deltas, color=colors, height=0.6)
        for yi, d in zip(y, deltas):
            offset = span * 0.05 if d >= 0 else -span * 0.05
            ax.text(d + offset, yi, f"{d:+.4f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8, color=TEXT)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlim(-span * 1.7, span * 1.0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_yticks(y, [_set_label(s) for s in part["feature_set"]])
        reference = table.loc[(table["model"] == model)
                              & (table["feature_set"] == REFERENCE_SET), "f1_macro"].iloc[0]
        ax.set_title(f"{MODEL_TITLES.get(model, model)}\nall features: F1-macro {reference:.4f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("change in F1-macro vs all features")
        ax.grid(axis="y", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Feature ablation: green = removing the features hurts (they help), "
                 "orange = removing them helps", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, FIGURE_NAME)


if __name__ == "__main__":
    import sys

    rows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run_ablation(train_rows=rows)
    print(result[["model", "feature_set", "f1_macro", "delta_f1_macro", "accuracy",
                  "train_seconds"]].to_string(index=False))
