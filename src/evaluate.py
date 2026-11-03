"""Metrics, confusion matrices, permutation importance and error analysis."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src import config, features
from src.eda import BLUE_RAMP, CLASS_COLORS, GRID, MUTED, TEXT, _save, _style

LABELS = sorted(config.COVER_TYPES)
SHORT = {1: "Spruce/Fir", 2: "Lodgepole", 3: "Ponderosa", 4: "Cottonwood",
         5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}


def score(model, X_test, y_test):
    """Quality metrics plus prediction time on the hold-out split."""
    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = time.perf_counter() - start
    per_class = f1_score(y_test, predictions, average=None, labels=LABELS)
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "f1_macro": round(f1_score(y_test, predictions, average="macro"), 4),
        "f1_weighted": round(f1_score(y_test, predictions, average="weighted"), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, predictions), 4),
        "predict_seconds": round(predict_seconds, 3),
        "predict_ms_per_1000": round(1000 * predict_seconds / len(X_test) * 1000, 3),
    }
    for label, value in zip(LABELS, per_class):
        metrics[f"f1_{SHORT[label]}"] = round(float(value), 4)
    return metrics, predictions


def report_table(y_true, y_pred):
    """Per-class precision/recall/F1/support as a DataFrame."""
    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=[config.COVER_TYPES[i] for i in LABELS],
        output_dict=True, zero_division=0,
    )
    return pd.DataFrame(report).T.round(4)


def plot_confusion(y_true, y_pred, title, filename):
    """Row-normalised confusion matrix (recall on the diagonal) + counts."""
    _style()
    counts = confusion_matrix(y_true, y_pred, labels=LABELS)
    shares = counts / counts.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(shares, cmap=BLUE_RAMP, vmin=0, vmax=100)
    names = [SHORT[i] for i in LABELS]
    ax.set_xticks(range(7), names, rotation=35, ha="right")
    ax.set_yticks(range(7), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    for i in range(7):
        for j in range(7):
            if counts[i, j] == 0:
                continue
            color = "white" if shares[i, j] > 55 else TEXT
            ax.text(j, i, f"{shares[i, j]:.1f}%\n{counts[i, j]:,}", ha="center",
                    va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="% of true class")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, filename), counts


def plot_model_comparison(results):
    """Quality vs cost: F1-macro, train time and model size side by side."""
    _style()
    order = results.sort_values("f1_macro")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    panels = [
        ("f1_macro", "F1-macro (test)", "{:.3f}"),
        ("accuracy", "Accuracy (test)", "{:.3f}"),
        ("train_seconds", "Training time, s (full train split)", "{:.1f}"),
        ("model_size_mb", "Model size, MB (pickled)", "{:.2f}"),
    ]
    for ax, (column, title, fmt) in zip(axes, panels):
        colors = ["#1c5cab" if m == results.loc[results["f1_macro"].idxmax(), "model"]
                  else "#86b6ef" for m in order["model"]]
        ax.barh(order["model"], order[column], color=colors, height=0.6)
        for i, v in enumerate(order[column]):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, order[column].max() * 1.3 if order[column].max() > 0 else 1)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Model comparison: quality vs cost (dark bar = best F1-macro)",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "11_model_comparison.png")


def plot_per_class_f1(results):
    """Grouped bars: F1 of every class for every model (imbalance effect)."""
    _style()
    models = results["model"].tolist()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    width = 0.8 / len(models)
    model_colors = ["#c3c2b7", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#6b6a63"]
    for k, model in enumerate(models):
        values = [results.loc[results["model"] == model, f"f1_{SHORT[c]}"].item() for c in LABELS]
        ax.bar(np.arange(7) + k * width - 0.4 + width / 2, values, width=width * 0.92,
               color=model_colors[k % len(model_colors)], label=model)
    ax.set_xticks(range(7), [SHORT[c] for c in LABELS])
    ax.set_ylabel("F1 on test split")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1: rare classes separate the models")
    ax.legend(ncol=len(models), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="x", visible=False)
    return _save(fig, "12_per_class_f1.png")


def permutation_importance_table(model, X_test, y_test, n=20_000, repeats=5):
    """Permutation importance on the raw input columns (drop in F1-macro).

    Permuting raw columns also breaks every engineered feature built from
    them, so the importance is reported for what the user actually enters.
    """
    X_sample, y_sample = features.stratified_subsample(X_test, y_test, n)
    result = permutation_importance(
        model, X_sample, y_sample, scoring="f1_macro", n_repeats=repeats,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    table = pd.DataFrame(
        {"feature": X_sample.columns, "importance_mean": result.importances_mean,
         "importance_std": result.importances_std}
    ).sort_values("importance_mean", ascending=False)
    return table.round(4)


def plot_importance(table, model_name):
    _style()
    table = table.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(table["feature"], table["importance_mean"], xerr=table["importance_std"],
            color="#2a78d6", height=0.6, error_kw={"ecolor": MUTED, "lw": 1})
    ax.set_xlabel("Drop in F1-macro when the column is shuffled")
    ax.set_title(f"Permutation importance ({model_name}, 20k test rows, 5 repeats)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "13_permutation_importance.png")


def error_analysis(model, X_test, y_test, predictions):
    """Where and how confidently the model is wrong.

    Returns a dict of DataFrames: errors by elevation band, by wilderness
    area, confidence of correct vs wrong predictions, and the 20 most
    confident mistakes.
    """
    frame = X_test.copy()
    frame["true"] = y_test.to_numpy()
    frame["pred"] = predictions
    frame["wrong"] = frame["true"] != frame["pred"]
    proba = model.predict_proba(X_test)
    frame["confidence"] = proba.max(axis=1)

    frame["elev_band"] = (frame["Elevation"] // 250) * 250
    by_band = frame.groupby("elev_band").agg(n=("wrong", "size"), error_rate=("wrong", "mean"))
    by_area = frame.groupby(config.WILDERNESS).agg(n=("wrong", "size"),
                                                   error_rate=("wrong", "mean"))
    by_area.index = [config.WILDERNESS_AREAS[int(i)] for i in by_area.index]

    confidence = frame.groupby("wrong")["confidence"].describe()[["mean", "25%", "50%", "75%"]]
    confidence.index = ["correct" if not w else "wrong" for w in confidence.index]

    confident_mistakes = frame[frame["wrong"]].sort_values("confidence", ascending=False).head(20)
    confident_mistakes = confident_mistakes.assign(
        true_name=confident_mistakes["true"].map(config.COVER_TYPES),
        pred_name=confident_mistakes["pred"].map(config.COVER_TYPES),
    )[["Elevation", config.WILDERNESS, config.SOIL, "true_name", "pred_name", "confidence"]]

    low_conf = frame["confidence"] < 0.6
    summary = {
        "test_rows": int(len(frame)),
        "errors": int(frame["wrong"].sum()),
        "share_low_confidence": round(float(low_conf.mean()), 4),
        "error_rate_low_confidence": round(float(frame.loc[low_conf, "wrong"].mean()), 4),
        "error_rate_high_confidence": round(float(frame.loc[~low_conf, "wrong"].mean()), 4),
    }
    return {
        "summary": summary,
        "by_elevation_band": by_band.round(4),
        "by_wilderness": by_area.round(4),
        "confidence": confidence.round(4),
        "confident_mistakes": confident_mistakes.round(3),
        "frame": frame,
    }


def plot_error_profile(analysis):
    """Error rate by elevation band with the number of test cells."""
    _style()
    band = analysis["by_elevation_band"]
    band = band[band["n"] >= 100]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(band.index.astype(int).astype(str), band["error_rate"] * 100, color="#2a78d6",
           width=0.7)
    for i, (rate, n) in enumerate(zip(band["error_rate"], band["n"])):
        ax.text(i, rate * 100, f"{rate * 100:.1f}%\nn={n:,}", ha="center", va="bottom",
                fontsize=7, color=TEXT)
    ax.set_xlabel("Elevation band start, m (250 m bands)")
    ax.set_ylabel("Error rate, %")
    ax.set_title("Best model: where are the errors? (test split)")
    ax.set_ylim(0, band["error_rate"].max() * 100 * 1.35)
    ax.grid(axis="x", visible=False)
    return _save(fig, "14_error_by_elevation.png")


def plot_learning_curve(curve):
    """F1-macro vs training-set size (justifies using the full data)."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"logreg": "#86b6ef", "random_forest": "#3987e5", "lightgbm": "#0d366b"}
    for model, part in curve.groupby("model"):
        ax.plot(part["train_rows"], part["f1_macro"], marker="o", markersize=5, linewidth=2,
                color=colors.get(model, MUTED), label=model)
        last = part.iloc[-1]
        ax.annotate(f"{last['f1_macro']:.3f}", (last["train_rows"], last["f1_macro"]),
                    textcoords="offset points", xytext=(6, 0), fontsize=8, color=TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("F1-macro on test split")
    ax.set_title("Learning curve: more data keeps helping the tree models")
    ax.legend(fontsize=8)
    return _save(fig, "15_learning_curve.png")


__all__ = [
    "score", "report_table", "plot_confusion", "plot_model_comparison", "plot_per_class_f1",
    "permutation_importance_table", "plot_importance", "error_analysis", "plot_error_profile",
    "plot_learning_curve", "CLASS_COLORS", "GRID",
]
