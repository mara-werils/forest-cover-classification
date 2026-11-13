"""Are the predicted probabilities trustworthy? Calibration and its repair.

The prediction form and the error analysis show the model's probability
("Aspen, 80 %"), so the number should mean what it says: among cells
predicted with confidence 0.8 about 80 % should be right. Measured on the
116k test cells for every model:

* **ECE** (expected calibration error, top label): cells are grouped into 15
  confidence bins, ECE = sum over bins of (share of cells) x |accuracy - mean
  confidence|. 0 = perfectly calibrated.
* **Brier score** (multi-class, sum over classes of squared error, 0..2) and
  **log-loss**: proper scoring rules that reward both calibration and
  sharpness.

Two repairs that do not touch the trained model:

* **Prior correction** for models trained with ``class_weight="balanced"``:
  the weights make the model learn p_w(k|x) proportional to p(k|x) / n_k, as
  if all classes were equally frequent. Multiplying by the training class
  frequencies n_k and renormalising undoes this exactly; no fitting needed.
* **Temperature scaling** (Guo et al., 2017): p_k proportional to
  p_k^(1/T), one parameter T fitted by minimising log-loss. T > 1 softens
  over-confident probabilities, T < 1 sharpens under-confident ones. T is
  fitted on one half of the test split and applied to the other half, then
  the halves swap (2-fold cross-fitting), so no cell is scored with a T
  fitted on itself.

Both repairs can change the arg-max, so accuracy and F1-macro are reported
for every variant as well: calibration must not be bought with quality.
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score

from src import config, data_loader, features, models
from src.eda import GRID, MUTED, TEXT, _save, _style
from src.evaluate import LABELS

N_BINS = 15
EPS = 1e-12
SELECTED = "lightgbm"
MODEL_NAMES = ("logreg", "random_forest", "hist_gb", "lightgbm", "lightgbm_unweighted")
TITLES = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
          "hist_gb": "HistGradientBoosting", "lightgbm": "LightGBM, balanced weights",
          "lightgbm_unweighted": "LightGBM, no weights"}
METHOD_TITLES = {"raw": "as trained", "temperature": "temperature scaled",
                 "prior_corrected": "prior corrected",
                 "prior_corrected+temperature": "prior corrected + temperature"}
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
FIGURE_NAME = "25_calibration.png"


def _path(name):
    """Output paths are built at call time, so tests can redirect REPORTS_DIR."""
    return config.REPORTS_DIR / name


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _index(y):
    return np.searchsorted(LABELS, np.asarray(y))


def reliability_bins(y_true, proba, n_bins=N_BINS):
    """Top-label reliability table: one row per non-empty confidence bin."""
    proba = np.asarray(proba)
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == _index(y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    # Bin i holds (edges[i], edges[i+1]]; confidence is never below 1/7.
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "cells": int(mask.sum()),
                         "mean_confidence": float(confidence[mask].mean()),
                         "accuracy": float(correct[mask].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, proba, n_bins=N_BINS):
    table = reliability_bins(y_true, proba, n_bins)
    weights = table["cells"] / table["cells"].sum()
    return float((weights * (table["accuracy"] - table["mean_confidence"]).abs()).sum())


def brier_score(y_true, proba):
    """Multi-class Brier score: mean over cells of sum_k (p_k - 1[y = k])^2."""
    proba = np.asarray(proba, dtype="float64")
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(proba)), _index(y_true)] = 1.0
    return float(((proba - one_hot) ** 2).sum(axis=1).mean())


def log_loss(y_true, proba):
    proba = np.asarray(proba, dtype="float64")
    picked = proba[np.arange(len(proba)), _index(y_true)]
    return float(-np.log(np.clip(picked, EPS, 1.0)).mean())


def calibration_metrics(y_true, proba):
    """Quality (arg-max) and calibration of one probability matrix."""
    proba = np.asarray(proba)
    predictions = np.asarray(LABELS)[proba.argmax(axis=1)]
    return {
        "accuracy": round(accuracy_score(y_true, predictions), 4),
        "f1_macro": round(f1_score(y_true, predictions, average="macro"), 4),
        "mean_confidence": round(float(proba.max(axis=1).mean()), 4),
        "ece": round(expected_calibration_error(y_true, proba), 4),
        "brier": round(brier_score(y_true, proba), 4),
        "log_loss": round(log_loss(y_true, proba), 4),
    }


# ---------------------------------------------------------------------------
# Repairs
# ---------------------------------------------------------------------------
def _normalise(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)


def prior_correction(proba, class_counts):
    """Undo balanced class weights: multiply by training class frequencies."""
    counts = np.asarray(class_counts, dtype="float64")
    return _normalise(np.asarray(proba, dtype="float64") * (counts / counts.sum()))


def apply_temperature(proba, temperature):
    """p_k ** (1 / T), renormalised (softmax of log-probabilities / T)."""
    logits = np.log(np.clip(np.asarray(proba, dtype="float64"), EPS, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    return _normalise(np.exp(logits))


def fit_temperature(y_true, proba):
    """T that minimises log-loss, searched on log T in [0.05, 20]."""
    result = minimize_scalar(lambda log_t: log_loss(y_true, apply_temperature(proba, np.exp(log_t))),
                             bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(result.x))


def cross_fit_temperature(y_true, proba, seed=config.RANDOM_STATE):
    """2-fold cross-fitting on the test split: each half gets the T of the other."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype="float64")
    halves = np.array_split(np.random.default_rng(seed).permutation(len(y_true)), 2)
    out = np.empty_like(proba)
    temperatures = []
    for fit_idx, apply_idx in ((halves[0], halves[1]), (halves[1], halves[0])):
        temperature = fit_temperature(y_true[fit_idx], proba[fit_idx])
        out[apply_idx] = apply_temperature(proba[apply_idx], temperature)
        temperatures.append(round(temperature, 3))
    return out, temperatures


def risk_coverage(y_true, proba, thresholds=THRESHOLDS):
    """If cells below a confidence threshold go to a human, how good is the rest?"""
    proba = np.asarray(proba)
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == _index(y_true)
    rows = []
    for threshold in thresholds:
        kept = confidence >= threshold
        rows.append({"threshold": threshold,
                     "coverage": round(float(kept.mean()), 4),
                     "accuracy_on_kept": round(float(correct[kept].mean()), 4) if kept.any() else None,
                     "share_of_errors_deferred": round(float((~correct & ~kept).sum()
                                                             / max((~correct).sum(), 1)), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probabilities of the compared models
# ---------------------------------------------------------------------------
def collect_probabilities(df=None, model_names=MODEL_NAMES):
    """Test-split probabilities of every model plus the training class counts.

    The selected model is loaded from disk; the others are refitted with their
    tuned params on the same 80/20 split as in step 5. Returns
    (y_test, {model: proba}, class_counts, {model: class_weight}).
    """
    df = df if df is not None else data_loader.load_parquet()
    if not config.TUNING_JSON.exists():
        raise FileNotFoundError(f"{config.TUNING_JSON} not found. Train the models first "
                                "(menu step 5).")
    tuning = json.loads(config.TUNING_JSON.read_text())
    X_train, X_test, y_train, y_test = features.split_data(df)
    counts = y_train.value_counts().reindex(LABELS, fill_value=0).to_numpy()
    probabilities, weights = {}, {}
    for name in model_names:
        start = time.perf_counter()
        base = "lightgbm" if name == "lightgbm_unweighted" else name
        params = dict(tuning.get(base, {}).get("best_params", {}))
        if name == "lightgbm_unweighted":
            params["class_weight"] = None
        if name == SELECTED and config.BEST_MODEL_PATH.exists():
            model = models.load_model()
        else:
            model, _ = models.fit_final(base, params, X_train, y_train)
        proba = model.predict_proba(X_test)
        # Align columns with LABELS (a class missing from a tiny train split).
        full = np.zeros((len(X_test), len(LABELS)))
        full[:, np.searchsorted(LABELS, model.classes_)] = proba
        probabilities[name] = full
        weights[name] = params.get("class_weight")
        print(f"[{name}] probabilities ready ({time.perf_counter() - start:.0f} s)")
    return y_test.to_numpy(), probabilities, counts, weights


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def variants(y_true, proba, class_counts, class_weight):
    """Probability matrices of every repair that applies to one model."""
    out = {"raw": (proba, None)}
    scaled, temps = cross_fit_temperature(y_true, proba)
    out["temperature"] = (scaled, temps)
    if class_weight == "balanced":
        corrected = prior_correction(proba, class_counts)
        out["prior_corrected"] = (corrected, None)
        scaled, temps = cross_fit_temperature(y_true, corrected)
        out["prior_corrected+temperature"] = (scaled, temps)
    return out


def verdicts(table, well_calibrated=0.01):
    """One sentence per model; ECE below ``well_calibrated`` needs no repair."""
    lines = []
    for model, part in table.groupby("model", sort=False):
        part = part.set_index("method")
        raw = part.loc["raw"]
        title = TITLES.get(model, model)
        if raw["ece"] < well_calibrated:
            lines.append(f"{title}: well calibrated as trained (confidence "
                         f"{raw['mean_confidence']:.3f} vs accuracy {raw['accuracy']:.3f}, "
                         f"ECE {raw['ece']:.4f}); no repair needed.")
            continue
        best_method = part["log_loss"].idxmin()
        best = part.loc[best_method]
        direction = ("under-confident" if raw["mean_confidence"] < raw["accuracy"]
                     else "over-confident")
        line = (f"{title}: {direction} as trained (confidence "
                f"{raw['mean_confidence']:.3f} vs accuracy {raw['accuracy']:.3f}, "
                f"ECE {raw['ece']:.4f}); best repair '{best_method}' -> ECE {best['ece']:.4f}, "
                f"log-loss {raw['log_loss']:.4f} -> {best['log_loss']:.4f}, "
                f"F1-macro {raw['f1_macro']:.4f} -> {best['f1_macro']:.4f}")
        if isinstance(best["temperatures"], list):
            line += f", T = {best['temperatures']}"
        lines.append(line + ".")
    return lines


def run_calibration(df=None, collected=None, save=True):
    """Metrics of every model and repair, reliability plot, risk-coverage table.

    ``collected`` takes the output of ``collect_probabilities`` (tests).
    About 10 minutes on 4 CPU threads, almost all of it refitting the models.
    """
    y_true, probabilities, counts, weights = collected or collect_probabilities(df)
    rows, curves, chosen = [], {}, None
    for name, proba in probabilities.items():
        for method, (matrix, temps) in variants(y_true, proba, counts, weights.get(name)).items():
            rows.append({"model": name, "method": method, **calibration_metrics(y_true, matrix),
                         "temperatures": temps})
            curves[(name, method)] = matrix
    table = pd.DataFrame(rows)
    lines = verdicts(table)
    print(table.drop(columns="temperatures").to_string(index=False))
    for line in lines:
        print(" * " + line)

    # Risk-coverage for the selected model with its best-calibrated probabilities.
    if SELECTED in probabilities:
        part = table[table["model"] == SELECTED].set_index("method")
        chosen = part["log_loss"].idxmin()
        coverage = risk_coverage(y_true, curves[(SELECTED, chosen)])
        coverage.insert(0, "method", chosen)
        print(coverage.to_string(index=False))
    else:
        coverage = pd.DataFrame()

    if save:
        table.to_csv(_path("calibration.csv"), index=False)
        coverage.to_csv(_path("risk_coverage.csv"), index=False)
        _path("calibration.json").write_text(json.dumps({
            "test_rows": len(y_true), "n_bins": N_BINS, "verdicts": lines,
            "selected_model_method": chosen,
            "results": table.to_dict(orient="records"),
            "risk_coverage": coverage.to_dict(orient="records"),
        }, indent=2))
        plot_calibration(y_true, curves, table)
    return table, coverage


def plot_calibration(y_true, curves, table, selected=SELECTED):
    """Left: reliability diagram. Right: ECE of every model, raw vs best repair."""
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6),
                                      gridspec_kw={"width_ratios": [1, 1.2]})
    left.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--",
              label="perfect calibration")
    styles = [((selected, "raw"), "#D08C2E", "o"),
              ((selected, "prior_corrected"), "#8FB996", "s"),
              ((selected, "temperature"), "#1F4D36", "D"),
              (("random_forest", "raw"), "#BDBDB7", "^"),
              (("logreg", "raw"), "#2a78d6", "v")]
    for key, color, marker in styles:
        if key not in curves:
            continue
        bins = reliability_bins(y_true, curves[key])
        bins = bins[bins["cells"] >= 50]           # tiny bins are pure noise
        left.plot(bins["mean_confidence"], bins["accuracy"], marker=marker, color=color,
                  markersize=4, linewidth=1.3,
                  label=f"{TITLES.get(key[0], key[0])}, {METHOD_TITLES[key[1]]}")
    left.set_xlim(0.25, 1.0)
    left.set_ylim(0.25, 1.0)
    left.set_xlabel("mean predicted confidence (bin)")
    left.set_ylabel("share of correct predictions (bin)")
    left.set_title("Reliability diagram (top label, 15 bins, bins with >= 50 cells)\n"
                   "below the diagonal = over-confident, above = under-confident",
                   fontsize=9, fontweight="bold")
    left.legend(frameon=False, fontsize=7, loc="upper left")

    names = list(dict.fromkeys(table["model"]))
    raw = table[table["method"] == "raw"].set_index("model")["ece"]
    best = table.loc[table.groupby("model")["log_loss"].idxmin()].set_index("model")
    y = np.arange(len(names))[::-1]
    right.barh(y + 0.18, [raw[n] for n in names], height=0.34, color="#D08C2E",
               label="as trained")
    right.barh(y - 0.18, [best.loc[n, "ece"] for n in names], height=0.34, color="#2E7D4F",
               label="best repair (lowest log-loss)")
    for yi, n in zip(y, names):
        right.text(raw[n] + 0.001, yi + 0.18, f"{raw[n]:.4f}", va="center", fontsize=7,
                   color=TEXT)
        right.text(best.loc[n, "ece"] + 0.001, yi - 0.18,
                   f"{best.loc[n, 'ece']:.4f}  ({METHOD_TITLES[best.loc[n, 'method']]})",
                   va="center", fontsize=7, color=TEXT)
    right.set_yticks(y, [TITLES.get(n, n) for n in names])
    right.set_xlim(0, max(raw.max(), best["ece"].max()) * 1.55)
    right.set_xlabel("expected calibration error (lower is better)")
    right.set_title("ECE before and after repair (repairs fitted without the scored cells)",
                    fontsize=9, fontweight="bold")
    right.legend(frameon=False, fontsize=7, loc="lower right")
    for ax in (left, right):
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    right.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, FIGURE_NAME)


if __name__ == "__main__":
    run_calibration()
