"""Extra checks of the selected approach.

1. Probability metrics (ROC-AUC, PR-AUC) for every compared model.
2. Group validation: leave-one-wilderness-area-out. The random split keeps
   neighbouring 30 m cells in both train and test, so it can be optimistic.
   Holding out a whole area shows how the model transfers to terrain it has
   never seen (the closest thing to a spatial split without coordinates).
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src import config, data_loader, features, models
from src.eda import GRID, MUTED, TEXT, _save, _style

PROB_CSV = config.REPORTS_DIR / "probability_metrics.csv"
GROUP_CSV = config.REPORTS_DIR / "group_validation.csv"
GROUP_JSON = config.REPORTS_DIR / "group_validation.json"
LABELS = sorted(config.COVER_TYPES)


def probability_metrics(model, X_test, y_test):
    """Macro one-vs-rest ROC-AUC and macro PR-AUC (average precision)."""
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    y_bin = label_binarize(y_test, classes=classes)
    return {
        "roc_auc_ovr_macro": round(roc_auc_score(y_test, proba, multi_class="ovr",
                                                 average="macro", labels=classes), 4),
        "pr_auc_macro": round(average_precision_score(y_bin, proba, average="macro"), 4),
    }


def run_probability_metrics(df=None):
    """Refit every compared model with its tuned params and add ROC/PR-AUC.

    The selected model is loaded from disk instead of being refitted.
    """
    df = df if df is not None else data_loader.load_parquet()
    X_train, X_test, y_train, y_test = features.split_data(df)
    tuning = json.loads(config.TUNING_JSON.read_text())
    best = json.loads(config.BEST_MODEL_META.read_text())
    variants = {name: (name, tuning[name]["best_params"]) for name in tuning}
    unweighted = dict(tuning["lightgbm"]["best_params"], class_weight=None)
    variants["lightgbm_unweighted"] = ("lightgbm", unweighted)

    rows = []
    for label, (name, params) in variants.items():
        start = time.perf_counter()
        if label == best["model"]:
            model = models.load_model()
        else:
            model, _ = models.fit_final(name, params, X_train, y_train)
        row = {"model": label, **probability_metrics(model, X_test, y_test)}
        rows.append(row)
        print(f"[{label}] ROC-AUC={row['roc_auc_ovr_macro']:.4f} "
              f"PR-AUC={row['pr_auc_macro']:.4f} ({time.perf_counter() - start:.0f} s)")
    table = pd.DataFrame(rows)
    table.to_csv(PROB_CSV, index=False)

    # Merge into the main results table so every view shows the new columns.
    results = pd.read_parquet(config.RESULTS_PARQUET)
    results = results.drop(columns=[c for c in table.columns if c != "model" and c in results])
    results = results.merge(table, on="model", how="left")
    results.to_parquet(config.RESULTS_PARQUET, index=False)
    results.to_csv(config.RESULTS_CSV, index=False)
    return table


def _area_scores(y_true, y_pred, seen):
    """Metrics on one held-out area; F1-macro over classes present there."""
    present = sorted(set(y_true))
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, labels=present, average="macro",
                                   zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "classes_in_area": len(present),
        "classes_unseen_in_train": [int(c) for c in present if c not in seen],
    }


ALL_GROUP_MODELS = ("baseline_dummy", "random_forest", "lightgbm", "lightgbm_no_area")


def leave_one_area_out(df=None, model_names=ALL_GROUP_MODELS, save=True):
    """Train on three wilderness areas, test on the fourth, for each area.

    "lightgbm_no_area" is the same LightGBM with the Wilderness column set to
    a constant: the held-out area's code was never seen in training, so this
    variant separates "unknown area code" from "different terrain".
    Compared with the selected model evaluated on that area's cells from the
    random 80/20 test split (random-split row of the table).
    """
    df = df if df is not None else data_loader.load_parquet()
    tuning = json.loads(config.TUNING_JSON.read_text())
    X = df[features.INPUT_COLS]
    y = df[config.TARGET].astype("int64")
    area = df[config.WILDERNESS]

    rows = []
    for area_id, area_name in config.WILDERNESS_AREAS.items():
        test_mask = (area == area_id).to_numpy()
        X_tr, y_tr = X[~test_mask], y[~test_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        seen = set(y_tr.unique())
        for name in model_names:
            start = time.perf_counter()
            Xa, Xb = X_tr, X_te
            if name == "lightgbm_no_area":
                Xa, Xb = X_tr.assign(**{config.WILDERNESS: 0}), X_te.assign(**{config.WILDERNESS: 0})
            if name == "baseline_dummy":
                model = DummyClassifier(strategy="most_frequent").fit(Xa, y_tr)
            else:
                base = "lightgbm" if name == "lightgbm_no_area" else name
                model, _ = models.fit_final(base, tuning[base]["best_params"], Xa, y_tr)
            predictions = model.predict(Xb)
            row = {"held_out_area": area_name, "model": name, "train_rows": len(X_tr),
                   "test_rows": len(X_te), **_area_scores(y_te, predictions, seen),
                   "seconds": round(time.perf_counter() - start, 1)}
            rows.append(row)
            print(f"[{area_name:15s}] {name:15s} F1-macro={row['f1_macro']:.4f} "
                  f"acc={row['accuracy']:.4f} unseen classes={row['classes_unseen_in_train']}")

    # Reference: the selected model on the same area under the random split.
    X_train, X_test, y_train, y_test = features.split_data(df)
    best = models.load_model()
    predictions = pd.Series(best.predict(X_test), index=X_test.index)
    for area_id, area_name in config.WILDERNESS_AREAS.items():
        idx = X_test.index[X_test[config.WILDERNESS] == area_id]
        rows.append({"held_out_area": area_name, "model": "lightgbm (random split)",
                     "train_rows": len(X_train), "test_rows": len(idx),
                     **_area_scores(y_test[idx], predictions[idx], set(y_train.unique())),
                     "seconds": 0.0})

    table = pd.DataFrame(rows)
    if save:
        table.to_csv(GROUP_CSV, index=False)
        GROUP_JSON.write_text(json.dumps(table.to_dict(orient="records"), indent=2))
    if save:
        plot_group_validation(table)
    return table


def plot_group_validation(table):
    """F1-macro per held-out area: random split vs unseen-area split."""
    _style()
    areas = list(config.WILDERNESS_AREAS.values())
    series = [("lightgbm (random split)", "Selected LightGBM, random 80/20 split", "#1F4D36"),
              ("lightgbm", "LightGBM, area held out", "#2E7D4F"),
              ("lightgbm_no_area", "LightGBM without area feature, area held out", "#D08C2E"),
              ("random_forest", "Random Forest, area held out", "#8FB996"),
              ("baseline_dummy", "Baseline, area held out", "#BDBDB7")]
    series = [item for item in series if item[0] in set(table["model"])]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.8 / len(series)
    x = np.arange(len(areas))
    for i, (model, label, color) in enumerate(series):
        values = [table.loc[(table["held_out_area"] == a) & (table["model"] == model),
                            "f1_macro"].iloc[0] for a in areas]
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width * 0.92, color=color,
                      label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}",
                    ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(x, areas)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1-macro on the held-out area")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Group validation: does the model transfer to a wilderness area it never saw?",
                 fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return _save(fig, "18_group_validation.png")


def run_robustness(df=None):
    """Both checks (about 25 minutes on 4 CPU threads)."""
    df = df if df is not None else data_loader.load_parquet()
    probability_table = run_probability_metrics(df)
    group_table = leave_one_area_out(df)
    return probability_table, group_table


if __name__ == "__main__":
    prob, group = run_robustness()
    print(prob.to_string(index=False))
    print(group.to_string(index=False))
