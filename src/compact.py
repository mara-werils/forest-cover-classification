"""How small can the model be? Size / speed / quality trade-off of LightGBM.

The selected LightGBM (255 leaves x 500 rounds x 7 classes) takes 91 MB and
~0.3-0.4 s per 1,000 cells. Its size grows linearly with the number of trees,
and cutting trees alone loses quality quickly (300 of 500 rounds: -40 % size,
F1-macro 0.950 instead of 0.956), so the number of leaves per tree is varied
as well.

Protocol (the test split is touched only once, at the end):

1. Split the training data 90/10 (stratified) into inner-train and validation.
2. Train LightGBM with 31, 63, 127 and 255 leaves with the recipe of the
   selected model (learning rate 0.05, balanced class weights, up to 1,000
   rounds) on inner-train and score every checkpoint (100 ... 1,000 rounds)
   on validation - one fit per leaf count, the checkpoints use
   ``num_iteration``.
3. Flag a leaf count as **diverged** when its validation F1-macro falls more
   than ``DIVERGENCE_DROP`` below its own earlier best, and exclude it.
   Among the rest choose the **smallest** checkpoint whose validation
   F1-macro is within ``TOLERANCE`` of the best one.
4. Train exactly that configuration on inner-train (boosting is deterministic,
   so it is the validated model, not a new refit) and compare it with the
   committed model on the test split: quality (paired bootstrap of F1-macro,
   McNemar), file size and prediction time. The committed model saw 10 % more
   training data, so the comparison is conservative.

Pilot (kept in ``reports/compact_final.json``): with learning rate 0.1 the
31-leaf model diverged on validation and the chosen 63 x 1000 configuration
diverged when refitted on the full training split (test F1-macro 0.43 after
0.954 on validation) - the same instability of large steps with balanced
class weights as the HistGradientBoosting run in ``diverged_runs.json``.
That is why the recipe stays at 0.05 and the validated model is not refitted.
"""

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src import config, data_loader, features, models, significance
from src.eda import GRID, MUTED, TEXT, _save, _style

LEAVES = (31, 63, 127, 255)
LEARNING_RATE = 0.05
MAX_ROUNDS = 1000
CHECKPOINTS = (100, 200, 300, 500, 700, 1000)
TOLERANCE = 0.002
DIVERGENCE_DROP = 0.01
PILOT = {
    "learning_rate": 0.1,
    "31_leaves": "validation F1-macro 0.846 after 200 rounds, 0.703 after 700: diverged",
    "63_leaves_x_1000_refit_on_full_train": "validation F1-macro 0.954, test F1-macro 0.429: "
                                             "diverged when refitted",
}
VALIDATION_SIZE = 0.10
FIGURE_NAME = "27_size_vs_quality.png"
COLORS = {31: "#BDBDB7", 63: "#8FB996", 127: "#2E7D4F", 255: "#1F4D36"}


def _path(name):
    """Output paths are built at call time, so tests can redirect REPORTS_DIR."""
    return config.REPORTS_DIR / name


def candidate_params(num_leaves, rounds=MAX_ROUNDS, learning_rate=LEARNING_RATE):
    return {"num_leaves": num_leaves, "n_estimators": rounds, "learning_rate": learning_rate,
            "min_child_samples": 10, "class_weight": "balanced"}


def booster_size_mb(booster, num_iteration=None):
    """Size of the model text (what LightGBM saves) for the first ``num_iteration`` rounds."""
    return len(booster.model_to_string(num_iteration=num_iteration)) / 1e6


def staged_scores(pipeline, X, y, checkpoints=CHECKPOINTS, timing_rows=20_000):
    """Quality, size and prediction time of the first k rounds, for every checkpoint k."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    booster = model.booster_
    transformed = prep.transform(X)
    timing = transformed[:timing_rows]
    rows = []
    for k in checkpoints:
        if k > booster.current_iteration():
            break
        predictions = model.classes_[booster.predict(transformed, num_iteration=k).argmax(axis=1)]
        start = time.perf_counter()
        booster.predict(timing, num_iteration=k)
        seconds = time.perf_counter() - start
        rows.append({"rounds": k,
                     "f1_macro": round(f1_score(y, predictions, average="macro"), 4),
                     "accuracy": round(accuracy_score(y, predictions), 4),
                     "size_mb": round(booster_size_mb(booster, k), 2),
                     "predict_ms_per_1000": round(1e6 * seconds / len(timing), 1)})
    return rows


def flag_divergence(table, drop=DIVERGENCE_DROP):
    """Mark every leaf count whose F1-macro falls ``drop`` below its own earlier best."""
    table = table.sort_values(["num_leaves", "rounds"]).copy()
    running_best = table.groupby("num_leaves")["f1_macro"].cummax()
    fell = (running_best - table["f1_macro"]) > drop
    diverged = fell.groupby(table["num_leaves"]).transform("any")
    table["diverged"] = diverged.astype(bool)
    return table.reset_index(drop=True)


def choose_compact(table, tolerance=TOLERANCE):
    """Smallest stable checkpoint whose validation F1-macro is within ``tolerance`` of the best."""
    if "diverged" in table:
        table = table[~table["diverged"]]
    if table.empty:
        raise ValueError("every configuration diverged; lower the learning rate")
    best = table["f1_macro"].max()
    eligible = table[table["f1_macro"] >= best - tolerance]
    return eligible.sort_values(["size_mb", "f1_macro"], ascending=[True, False]).iloc[0]


def pareto_front(table):
    """Checkpoints not dominated by another one that is both smaller and better."""
    keep = []
    for _, row in table.iterrows():
        dominated = ((table["size_mb"] <= row["size_mb"]) & (table["f1_macro"] >= row["f1_macro"])
                     & ((table["size_mb"] < row["size_mb"])
                        | (table["f1_macro"] > row["f1_macro"]))).any()
        keep.append(not dominated)
    return table[keep].sort_values("size_mb")


REFERENCE = (255, 500)          # the selected configuration
BUDGETS_MB = (10, 25, 50)


def budget_table(curve, budgets=BUDGETS_MB, reference=REFERENCE):
    """Best stable validation F1-macro under each size budget, vs the selected configuration."""
    stable = curve[~curve["diverged"]] if "diverged" in curve else curve
    ref = stable[(stable["num_leaves"] == reference[0]) & (stable["rounds"] == reference[1])]
    ref_f1 = float(ref["f1_macro"].iloc[0]) if len(ref) else float(stable["f1_macro"].max())
    ref_size = float(ref["size_mb"].iloc[0]) if len(ref) else float(stable["size_mb"].max())
    rows = []
    for budget in budgets:
        fits = stable[stable["size_mb"] <= budget]
        if fits.empty:
            continue
        best = fits.loc[fits["f1_macro"].idxmax()]
        rows.append({"budget_mb": budget, "num_leaves": int(best["num_leaves"]),
                     "rounds": int(best["rounds"]), "size_mb": float(best["size_mb"]),
                     "f1_macro": float(best["f1_macro"]),
                     "f1_change_vs_selected_config": round(float(best["f1_macro"]) - ref_f1, 4),
                     "size_vs_selected_config": round(float(best["size_mb"]) / ref_size, 3),
                     "predict_ms_per_1000": float(best["predict_ms_per_1000"])})
    return pd.DataFrame(rows), {"num_leaves": reference[0], "rounds": reference[1],
                                "f1_macro": ref_f1, "size_mb": ref_size}


def describe_speed(ratio):
    """'2.0x faster' or '1.7x slower' from selected_time / compact_time."""
    return f"{ratio:.1f}x faster" if ratio >= 1 else f"{1 / ratio:.1f}x slower"


def inner_split(X_train, y_train):
    """Step 1: stratified 90/10 split of the training data (inner-train, validation)."""
    return train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, stratify=y_train,
                            random_state=config.RANDOM_STATE)


def search(X_inner, X_val, y_inner, y_val, leaves=LEAVES, rounds=MAX_ROUNDS,
           checkpoints=CHECKPOINTS):
    """Step 2-3: validation curve of every leaf count, with the divergence flag."""
    rows = []
    for num_leaves in leaves:
        start = time.perf_counter()
        pipeline, cost = models.fit_final("lightgbm", candidate_params(num_leaves, rounds),
                                          X_inner, y_inner)
        for row in staged_scores(pipeline, X_val, y_val, checkpoints):
            rows.append({"num_leaves": num_leaves, **row})
        print(f"[{num_leaves:3d} leaves] fitted in {cost['train_seconds']:.0f} s, "
              f"best validation F1-macro "
              f"{max(r['f1_macro'] for r in rows if r['num_leaves'] == num_leaves):.4f} "
              f"({time.perf_counter() - start:.0f} s with scoring)")
    return flag_divergence(pd.DataFrame(rows))


def compare_on_test(compact, reference, X_test, y_test, n_boot=significance.N_BOOT):
    """Step 4: the refitted compact model against the committed one on the test split."""
    rows = {}
    predictions = {}
    for name, pipeline in (("selected", reference), ("compact", compact)):
        start = time.perf_counter()
        predictions[name] = pipeline.predict(X_test)
        seconds = time.perf_counter() - start
        rows[name] = {
            "f1_macro": round(f1_score(y_test, predictions[name], average="macro"), 4),
            "accuracy": round(accuracy_score(y_test, predictions[name]), 4),
            "size_mb": round(models.model_size_mb(pipeline), 2),
            "predict_ms_per_1000": round(1e6 * seconds / len(X_test), 1),
        }
    paired = significance.paired_bootstrap(y_test, predictions["compact"],
                                           predictions["selected"], n_boot)
    test = significance.mcnemar(y_test, predictions["compact"], predictions["selected"])
    return rows, {**{k: round(v, 4) for k, v in paired.items()}, **test}


def run_compact(df=None, leaves=LEAVES, rounds=MAX_ROUNDS, checkpoints=CHECKPOINTS,
                save=True, save_model=False):
    """Search on validation, train the chosen size, compare on test (~20-30 min).

    ``save_model=True`` also writes the compact pipeline to
    ``models/compact_model.joblib`` (git-ignored, not used by the app).
    """
    df = df if df is not None else data_loader.load_parquet()
    X_train, X_test, y_train, y_test = features.split_data(df)
    X_inner, X_val, y_inner, y_val = inner_split(X_train, y_train)
    curve = search(X_inner, X_val, y_inner, y_val, leaves, rounds, checkpoints)
    chosen = choose_compact(curve)
    print(curve.to_string(index=False))
    diverged = sorted(int(n) for n in curve.loc[curve["diverged"], "num_leaves"].unique())
    if diverged:
        print(f"Diverged and excluded: {diverged} leaves")
    print(f"\nChosen on validation: {int(chosen['num_leaves'])} leaves x "
          f"{int(chosen['rounds'])} rounds ({chosen['size_mb']} MB, "
          f"F1-macro {chosen['f1_macro']:.4f})")

    params = candidate_params(int(chosen["num_leaves"]), int(chosen["rounds"]))
    compact, cost = models.fit_final("lightgbm", params, X_inner, y_inner)
    reference = models.load_model()
    final, paired = compare_on_test(compact, reference, X_test, y_test)
    final["compact"]["train_seconds"] = cost["train_seconds"]
    final["compact"]["trained_on_rows"] = len(X_inner)
    final["selected"]["trained_on_rows"] = len(X_train)
    shrink = final["compact"]["size_mb"] / final["selected"]["size_mb"]
    speed = final["selected"]["predict_ms_per_1000"] / final["compact"]["predict_ms_per_1000"]
    significant = paired["delta_ci_low"] > 0 or paired["delta_ci_high"] < 0
    verdict = (f"Compact LightGBM ({params['num_leaves']} leaves x {params['n_estimators']} "
               f"rounds): {shrink:.1%} of the size, {describe_speed(speed)} prediction, "
               f"F1-macro {final['compact']['f1_macro']:.4f} vs {final['selected']['f1_macro']:.4f} "
               f"(paired 95 % CI of the difference [{paired['delta_ci_low']:+.4f}, "
               f"{paired['delta_ci_high']:+.4f}] -> "
               f"{'significant' if significant else 'not significant'}).")
    budgets, reference_row = budget_table(curve)
    print(json.dumps(final, indent=2))
    print(" * " + verdict)
    print("Best stable configuration per size budget (validation):")
    print(budgets.to_string(index=False))

    if save:
        curve.to_csv(_path("compact_candidates.csv"), index=False)
        _path("compact_final.json").write_text(json.dumps({
            "learning_rate": LEARNING_RATE, "validation_size": VALIDATION_SIZE,
            "tolerance": TOLERANCE, "divergence_drop": DIVERGENCE_DROP,
            "diverged_leaf_counts": diverged,
            "chosen_params": params, "test": final, "compact_minus_selected": paired,
            "verdict": verdict,
            "selected_config_on_validation": reference_row,
            "size_budgets_validation": budgets.to_dict(orient="records"),
            "pareto_front_validation": pareto_front(curve[~curve["diverged"]]).to_dict(
                orient="records"),
            "pilot_learning_rate_0_1": PILOT,
        }, indent=2, default=float))
        plot_tradeoff(curve, chosen, final)
    if save_model:
        joblib.dump(compact, config.MODELS_DIR / "compact_model.joblib", compress=3)
    return curve, final, paired


def plot_tradeoff(curve, chosen, final):
    """Validation F1-macro against model size, one line per leaf count."""
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for num_leaves, part in curve.groupby("num_leaves"):
        unstable = bool(part["diverged"].any()) if "diverged" in part else False
        ax.plot(part["size_mb"], part["f1_macro"], marker="o", markersize=4, linewidth=1.4,
                linestyle="--" if unstable else "-", color=COLORS.get(num_leaves, MUTED),
                label=f"{num_leaves} leaves" + (" (diverged, excluded)" if unstable else ""))
        last = part.iloc[-1]
        ax.text(last["size_mb"] * 1.04, last["f1_macro"], f"{int(last['rounds'])} rounds",
                fontsize=7, color=MUTED, va="center")
    ax.scatter([chosen["size_mb"]], [chosen["f1_macro"]], s=160, facecolors="none",
               edgecolors="#D08C2E", linewidths=2, zorder=4,
               label=f"chosen: {int(chosen['num_leaves'])} leaves x {int(chosen['rounds'])} rounds")
    stable = curve[~curve["diverged"]] if "diverged" in curve else curve
    ref = stable[(stable["num_leaves"] == REFERENCE[0]) & (stable["rounds"] == REFERENCE[1])]
    if len(ref):
        ax.scatter(ref["size_mb"], ref["f1_macro"], marker="*", s=180, color="#2a78d6",
                   zorder=5, label="selected configuration (255 x 500), on 90 % of train")
    best = stable["f1_macro"].max()
    ax.axhspan(best - TOLERANCE, best, color="#D08C2E", alpha=0.08,
               label=f"within {TOLERANCE} of the best")
    ax.set_xscale("log")
    ax.set_xlabel("model size, MB (log scale)")
    ax.set_ylabel("F1-macro on the validation split")
    sel, comp = final["selected"], final["compact"]
    ax.set_title(f"Size vs quality. Test: selected {sel['size_mb']:.0f} MB, F1 {sel['f1_macro']:.4f}"
                 f"  |  compact {comp['size_mb']:.0f} MB, F1 {comp['f1_macro']:.4f}",
                 fontsize=9, fontweight="bold")
    low = stable["f1_macro"].quantile(0.15)
    ax.set_ylim(low, best + 0.004)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.text(0.01, 0.97, "each point = first k rounds of one fitted model", transform=ax.transAxes,
            fontsize=7, color=TEXT, va="top")
    fig.tight_layout()
    return _save(fig, FIGURE_NAME)


if __name__ == "__main__":
    run_compact(save_model=True)
