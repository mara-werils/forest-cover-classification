"""Is the selected model really better? Confidence intervals and paired tests.

The models in step 5 differ by a few thousandths of F1-macro (Random Forest
0.949, HistGradientBoosting 0.955, LightGBM 0.956), so the ranking may be
noise of one particular test split. Two complementary checks on the same
116k test cells:

* **Bootstrap 95 % confidence interval of F1-macro** for every model, and a
  **paired bootstrap** of the difference between two models (both models are
  scored on the same resampled cells, so the shared difficulty of a cell
  cancels out - much tighter than comparing two separate intervals).
* **McNemar's test** for every pair: it looks only at the cells where exactly
  one of the two models is right (b = only A right, c = only B right) and
  tests b = c with an exact binomial test. It compares accuracy, not
  F1-macro, so it is a second, independent view. p-values are corrected for
  the number of pairs with the Holm method.

Speed-up used for the bootstrap: F1-macro depends on the predictions only
through the 7 x 7 confusion matrix, and resampling n cells with replacement is
exactly a multinomial draw of n counts over the matrix cells. Drawing the 49
(or, for a pair, 7^3 = 343) cell counts instead of 116k indices gives the same
distribution at a fraction of the cost.
"""

import json
import time
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config, data_loader, features, models
from src.eda import GRID, MUTED, TEXT, _save, _style
from src.evaluate import LABELS

K = len(LABELS)
N_BOOT = 2000
ALPHA = 0.05
SELECTED = "lightgbm"
# Models compared in step 5 (the baseline gets an interval but is not paired).
MODEL_NAMES = ("baseline_dummy", "logreg", "random_forest", "hist_gb", "lightgbm",
               "lightgbm_unweighted")
TITLES = {"baseline_dummy": "Baseline (most frequent)", "logreg": "Logistic Regression",
          "random_forest": "Random Forest", "hist_gb": "HistGradientBoosting",
          "lightgbm": "LightGBM, balanced weights", "lightgbm_unweighted": "LightGBM, no weights"}
FIGURE_NAME = "24_model_significance.png"


def _path(name):
    """Output paths are built at call time, so tests can redirect REPORTS_DIR."""
    return config.REPORTS_DIR / name


# ---------------------------------------------------------------------------
# F1-macro from confusion matrices (vectorised over bootstrap replicates)
# ---------------------------------------------------------------------------
def encode(labels):
    """Cover types 1..7 -> indices 0..6."""
    return np.searchsorted(LABELS, np.asarray(labels))


def confusion_counts(y_true, y_pred):
    """7 x 7 confusion matrix (rows = true class) as plain counts."""
    codes = encode(y_true) * K + encode(y_pred)
    return np.bincount(codes, minlength=K * K).reshape(K, K)


def f1_macro_from_confusion(confusion):
    """F1-macro of one matrix (K, K) or of a batch (B, K, K).

    Per class F1 = 2 TP / (2 TP + FP + FN); a class that is neither present
    nor predicted gets 0, as in scikit-learn with zero_division=0.
    """
    confusion = np.asarray(confusion, dtype="float64")
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    fn = confusion.sum(axis=-1) - tp
    fp = confusion.sum(axis=-2) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1.mean(axis=-1)


# ---------------------------------------------------------------------------
# Bootstrap and McNemar
# ---------------------------------------------------------------------------
def bootstrap_f1(y_true, y_pred, n_boot=N_BOOT, seed=config.RANDOM_STATE, alpha=ALPHA):
    """Point F1-macro and its percentile bootstrap (1 - alpha) interval."""
    counts = confusion_counts(y_true, y_pred).ravel()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(counts.sum(), counts / counts.sum(), size=n_boot)
    scores = f1_macro_from_confusion(draws.reshape(n_boot, K, K))
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {"f1_macro": float(f1_macro_from_confusion(counts.reshape(K, K))),
            "ci_low": float(low), "ci_high": float(high),
            "ci_width": float(high - low)}


def paired_bootstrap(y_true, pred_a, pred_b, n_boot=N_BOOT, seed=config.RANDOM_STATE,
                     alpha=ALPHA):
    """Bootstrap of F1-macro(A) - F1-macro(B) on the same resampled cells."""
    codes = (encode(y_true) * K + encode(pred_a)) * K + encode(pred_b)
    counts = np.bincount(codes, minlength=K ** 3)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(counts.sum(), counts / counts.sum(), size=n_boot)
    cube = draws.reshape(n_boot, K, K, K)            # (replicate, true, pred A, pred B)
    diff = (f1_macro_from_confusion(cube.sum(axis=3))
            - f1_macro_from_confusion(cube.sum(axis=2)))
    cells = counts.reshape(K, K, K)                  # (true, pred A, pred B)
    point = (f1_macro_from_confusion(cells.sum(axis=2))
             - f1_macro_from_confusion(cells.sum(axis=1)))
    low, high = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return {"delta_f1_macro": float(point), "delta_ci_low": float(low),
            "delta_ci_high": float(high),
            # Share of replicates where A is not better than B (one-sided, informal).
            "share_a_not_better": float((diff <= 0).mean())}


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on the cells where exactly one model is right."""
    right_a = np.asarray(pred_a) == np.asarray(y_true)
    right_b = np.asarray(pred_b) == np.asarray(y_true)
    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    p_value = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"only_a_right": b, "only_b_right": c, "mcnemar_p": float(p_value)}


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p_values = np.asarray(p_values, dtype="float64")
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


# ---------------------------------------------------------------------------
# Predictions of the compared models on the test split
# ---------------------------------------------------------------------------
def collect_predictions(df=None, model_names=MODEL_NAMES):
    """Test-split predictions of every model, refitted with its tuned params.

    The selected model is loaded from disk (exactly the committed model);
    the others are refitted on the same 80/20 split as in step 5.
    Returns a DataFrame: y_true + one column per model, indexed like the test split.
    """
    df = df if df is not None else data_loader.load_parquet()
    if not config.TUNING_JSON.exists():
        raise FileNotFoundError(f"{config.TUNING_JSON} not found. Train the models first "
                                "(menu step 5).")
    tuning = json.loads(config.TUNING_JSON.read_text())
    X_train, X_test, y_train, y_test = features.split_data(df)
    frame = pd.DataFrame({"y_true": y_test.astype("int8")}, index=X_test.index)
    for name in model_names:
        start = time.perf_counter()
        if name == SELECTED and config.BEST_MODEL_PATH.exists():
            model = models.load_model()
        else:
            base = "lightgbm" if name == "lightgbm_unweighted" else name
            params = dict(tuning.get(base, {}).get("best_params", {}))
            if name == "lightgbm_unweighted":
                params["class_weight"] = None
            model, _ = models.fit_final(base, params, X_train, y_train)
        frame[name] = model.predict(X_test).astype("int8")
        print(f"[{name}] predictions ready ({time.perf_counter() - start:.0f} s)")
    return frame


# ---------------------------------------------------------------------------
# Tables, verdicts and the full run
# ---------------------------------------------------------------------------
def model_table(predictions, n_boot=N_BOOT):
    rows = []
    for name in [c for c in predictions.columns if c != "y_true"]:
        rows.append({"model": name,
                     **bootstrap_f1(predictions["y_true"], predictions[name], n_boot)})
    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    return table.round(4).reset_index(drop=True)


def pair_table(predictions, n_boot=N_BOOT, exclude=("baseline_dummy",)):
    names = [c for c in predictions.columns if c not in ("y_true", *exclude)]
    # Order inside a pair: the model with the higher F1-macro first (A).
    score = {n: bootstrap_f1(predictions["y_true"], predictions[n], n_boot=1)["f1_macro"]
             for n in names}
    rows = []
    for one, two in combinations(names, 2):
        a, b = (one, two) if score[one] >= score[two] else (two, one)
        y = predictions["y_true"]
        rows.append({"model_a": a, "model_b": b,
                     **paired_bootstrap(y, predictions[a], predictions[b], n_boot),
                     **mcnemar(y, predictions[a], predictions[b])})
    table = pd.DataFrame(rows)
    table["mcnemar_p_holm"] = holm(table["mcnemar_p"])
    table["f1_difference_significant"] = (table["delta_ci_low"] > 0) | (table["delta_ci_high"] < 0)
    table["accuracy_difference_significant"] = table["mcnemar_p_holm"] < ALPHA
    return table.round({"delta_f1_macro": 4, "delta_ci_low": 4, "delta_ci_high": 4,
                        "share_a_not_better": 4})


def verdicts(pairs, selected=SELECTED):
    """One sentence per pair that involves the selected model."""
    lines = []
    for _, row in pairs.iterrows():
        if selected not in (row["model_a"], row["model_b"]):
            continue
        f1_word = "significant" if row["f1_difference_significant"] else "NOT significant"
        acc_word = "significant" if row["accuracy_difference_significant"] else "not significant"
        lines.append(
            f"{TITLES.get(row['model_a'], row['model_a'])} vs "
            f"{TITLES.get(row['model_b'], row['model_b'])}: "
            f"dF1-macro {row['delta_f1_macro']:+.4f} "
            f"[{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}] -> {f1_word}; "
            f"McNemar {row['only_a_right']} vs {row['only_b_right']} cells, "
            f"Holm p = {row['mcnemar_p_holm']:.2g} -> accuracy difference {acc_word}.")
    return lines


def run_significance(df=None, predictions=None, n_boot=N_BOOT, save=True):
    """Collect (or take) test predictions, then intervals, paired tests and plots.

    About 15 minutes on 4 CPU threads, almost all of it refitting the models.
    """
    if predictions is None:
        predictions = collect_predictions(df)
    if save:
        predictions.to_parquet(_path("test_predictions.parquet"))
    models_ = model_table(predictions, n_boot)
    pairs = pair_table(predictions, n_boot)
    lines = verdicts(pairs)
    print(models_.to_string(index=False))
    for line in lines:
        print(" * " + line)
    if save:
        models_.to_csv(_path("significance_models.csv"), index=False)
        pairs.to_csv(_path("significance_pairs.csv"), index=False)
        _path("significance.json").write_text(json.dumps({
            "test_rows": len(predictions), "n_bootstrap": n_boot, "alpha": ALPHA,
            "verdicts": lines,
            "models": models_.to_dict(orient="records"),
            "pairs": pairs.to_dict(orient="records"),
        }, indent=2, default=bool))
        plot_significance(models_, pairs)
    return models_, pairs


def plot_significance(models_, pairs, selected=SELECTED):
    """Left: F1-macro with 95 % CI. Right: paired differences to the selected model."""
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.9),
                                      gridspec_kw={"width_ratios": [1, 1.15]})
    strong = models_[models_["f1_macro"] > 0.8].iloc[::-1]
    weak = models_[models_["f1_macro"] <= 0.8]
    y = np.arange(len(strong))
    colors = ["#1F4D36" if m == selected else "#8FB996" for m in strong["model"]]
    left.errorbar(strong["f1_macro"], y,
                  xerr=[strong["f1_macro"] - strong["ci_low"], strong["ci_high"] - strong["f1_macro"]],
                  fmt="none", ecolor=MUTED, capsize=4, linewidth=1.2)
    left.scatter(strong["f1_macro"], y, color=colors, s=45, zorder=3)
    for yi, (_, row) in zip(y, strong.iterrows()):
        left.text(row["ci_high"] + 0.0006, yi, f"{row['f1_macro']:.4f} "
                  f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]", va="center", fontsize=7.5,
                  color=TEXT)
    left.set_yticks(y, [TITLES.get(m, m) for m in strong["model"]])
    lo, hi = strong["ci_low"].min(), strong["ci_high"].max()
    left.set_xlim(lo - 0.002, hi + 0.012)
    note = ", ".join(f"{TITLES.get(r.model, r.model)} {r.f1_macro:.3f}" for r in weak.itertuples())
    left.set_title("F1-macro with 95 % bootstrap interval" + (f"\n(off scale: {note})" if note else ""),
                   fontsize=9, fontweight="bold")
    left.set_xlabel("F1-macro on the 116k test cells")

    mine = pairs[(pairs["model_a"] == selected) | (pairs["model_b"] == selected)].copy()
    # Express every pair as selected minus other.
    sign = np.where(mine["model_a"] == selected, 1, -1)
    mine["other"] = np.where(sign == 1, mine["model_b"], mine["model_a"])
    mine["d"] = sign * mine["delta_f1_macro"]
    mine["lo"] = np.where(sign == 1, mine["delta_ci_low"], -mine["delta_ci_high"])
    mine["hi"] = np.where(sign == 1, mine["delta_ci_high"], -mine["delta_ci_low"])
    # Weak models (off the left scale) would squash the interesting differences.
    mine = mine[mine["other"].isin(set(strong["model"]))].sort_values("d")
    y = np.arange(len(mine))
    sig = (mine["lo"] > 0) | (mine["hi"] < 0)
    right.errorbar(mine["d"], y, xerr=[mine["d"] - mine["lo"], mine["hi"] - mine["d"]],
                   fmt="none", ecolor=MUTED, capsize=4, linewidth=1.2)
    right.scatter(mine["d"], y, color=np.where(sig, "#2E7D4F", "#D08C2E"), s=45, zorder=3)
    right.axvline(0, color=MUTED, linewidth=0.8)
    for yi, (_, row), s in zip(y, mine.iterrows(), sig):
        right.text(row["hi"] + 0.0004, yi, f"{row['d']:+.4f}" + ("" if s else "  (n.s.)"),
                   va="center", fontsize=7.5, color=TEXT)
    right.set_yticks(y, [f"vs {TITLES.get(o, o)}" for o in mine["other"]])
    right.set_xlim(min(mine["lo"].min(), 0) - 0.003, mine["hi"].max() + 0.01)
    right.set_title("Paired difference: selected LightGBM minus other model\n"
                    "green = interval excludes 0, orange = not significant",
                    fontsize=9, fontweight="bold")
    right.set_xlabel("difference in F1-macro (paired bootstrap, 95 %)")
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    return _save(fig, FIGURE_NAME)


if __name__ == "__main__":
    run_significance()
