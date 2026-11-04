"""Model-free feature relevance: how much does each input tell about the class?

Three complementary measures, all computed before any model is trained:

* Kruskal-Wallis H for every numeric feature (rank-based ANOVA, no
  normality assumption) with the epsilon-squared effect size
  eps2 = H / (n - 1) (Tomczak & Tomczak, 2014), which - unlike the
  p-value - does not explode with 581k rows;
* bias-corrected Cramer's V (Bergsma, 2013) for the two categorical
  features, Wilderness and Soil;
* mutual information I(feature; class) in bits on a stratified
  subsample, the one measure that is comparable across numeric and
  categorical inputs.

The ranking is then compared with the permutation importance of the
trained model: features that are strong alone but unimportant for the
model are redundant, features that are weak alone but important are
used through interactions.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from src import config, features
from src.eda import MUTED, TEXT, _save, _style

RELEVANCE_CSV = config.REPORTS_DIR / "feature_relevance.csv"
RELEVANCE_JSON = config.REPORTS_DIR / "feature_relevance.json"
CATEGORICAL = [config.WILDERNESS, config.SOIL]
MI_SAMPLE = 60_000


def kruskal_effect_size(values, labels):
    """Kruskal-Wallis H, p-value and epsilon-squared for one numeric column."""
    values = np.asarray(values)
    labels = np.asarray(labels)
    groups = [values[labels == c] for c in np.unique(labels)]
    groups = [g for g in groups if len(g) > 0]
    h, p = stats.kruskal(*groups)
    # Share of rank variance explained by the class; perfect separation of
    # k equal groups gives 1 - 1/k^2, not 1 (within-group ranks still vary).
    epsilon2 = h / (len(values) - 1)
    return {"statistic": float(h), "p_value": float(p), "effect_size": float(epsilon2)}


def cramers_v(x, y):
    """Bias-corrected Cramer's V between two categorical arrays (0 = none, 1 = perfect)."""
    table = pd.crosstab(np.asarray(x), np.asarray(y)).to_numpy()
    n = table.sum()
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    r, c = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (c - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denominator = min(r_corr - 1, c_corr - 1)
    v = float(np.sqrt(phi2_corr / denominator)) if denominator > 0 else 0.0
    return {"statistic": float(chi2), "p_value": float(p), "effect_size": v}


def mutual_information(df, columns, n=MI_SAMPLE, seed=config.RANDOM_STATE):
    """I(column; class) in bits on a stratified subsample.

    Categorical columns are treated as discrete, numeric ones with the
    k-NN estimator of scikit-learn (Kraskov et al.).
    """
    X, y = features.stratified_subsample(df[columns], df[config.TARGET], n, seed=seed)
    discrete = [col in CATEGORICAL for col in columns]
    nats = mutual_info_classif(X.to_numpy(dtype="float64"), y.to_numpy(),
                               discrete_features=discrete, random_state=seed)
    return dict(zip(columns, nats / np.log(2)))


def class_entropy_bits(labels):
    """Entropy of the target: the upper bound of any mutual information."""
    shares = pd.Series(labels).value_counts(normalize=True).to_numpy()
    return float(-(shares * np.log2(shares)).sum())


def relevance_table(df, mi_sample=MI_SAMPLE):
    """One row per input feature with its test, effect size and MI."""
    labels = df[config.TARGET].to_numpy()
    rows = []
    for col in config.NUMERIC_COLS:
        result = kruskal_effect_size(df[col].to_numpy(), labels)
        rows.append({"feature": col, "kind": "numeric", "test": "Kruskal-Wallis",
                     "effect_measure": "epsilon_squared", **result})
    for col in CATEGORICAL:
        result = cramers_v(df[col].to_numpy(), labels)
        rows.append({"feature": col, "kind": "categorical", "test": "chi-square",
                     "effect_measure": "cramers_v", **result})
    table = pd.DataFrame(rows)
    mi = mutual_information(df, config.NUMERIC_COLS + CATEGORICAL, n=mi_sample)
    table["mutual_info_bits"] = table["feature"].map(mi)
    table["mi_share_of_class_entropy"] = table["mutual_info_bits"] / class_entropy_bits(labels)
    table = table.sort_values("mutual_info_bits", ascending=False).reset_index(drop=True)
    table["mi_rank"] = np.arange(1, len(table) + 1)
    return table


def compare_with_permutation_importance(table, importance):
    """Spearman rank correlation of MI with the model's permutation importance."""
    merged = table.merge(importance[["feature", "importance_mean"]], on="feature", how="inner")
    if len(merged) < 3:
        return None, merged
    rho, p = stats.spearmanr(merged["mutual_info_bits"], merged["importance_mean"])
    merged["importance_rank"] = merged["importance_mean"].rank(ascending=False).astype(int)
    merged["rank_gap"] = merged["mi_rank"] - merged["importance_rank"]
    return {"spearman_rho": round(float(rho), 3), "p_value": round(float(p), 4),
            "features": len(merged)}, merged


def plot_relevance(table, merged=None):
    """Left: MI as % of class entropy. Right: MI vs model importance."""
    _style()
    ordered = table.sort_values("mutual_info_bits")
    fig, axes = plt.subplots(1, 2 if merged is not None else 1, figsize=(15, 5),
                             squeeze=False)
    ax = axes[0, 0]
    colors = ["#1c5cab" if k == "categorical" else "#86b6ef" for k in ordered["kind"]]
    ax.barh(ordered["feature"], 100 * ordered["mi_share_of_class_entropy"], color=colors,
            height=0.6)
    for i, (share, effect, measure) in enumerate(zip(
            ordered["mi_share_of_class_entropy"], ordered["effect_size"],
            ordered["effect_measure"])):
        label = "V" if measure == "cramers_v" else "eps2"
        ax.text(100 * share, i, f" {100 * share:.1f}%  ({label}={effect:.2f})", va="center",
                fontsize=8, color=TEXT)
    ax.set_xlabel("Mutual information with the cover type, % of class entropy")
    ax.set_xlim(0, 100 * ordered["mi_share_of_class_entropy"].max() * 1.45)
    ax.set_title("Univariate relevance (dark = categorical)")
    ax.grid(axis="y", visible=False)

    if merged is not None:
        ax = axes[0, 1]
        ax.scatter(merged["mutual_info_bits"], merged["importance_mean"], s=40,
                   color="#2a78d6", zorder=3)
        weak = (merged["importance_mean"] < 0.05) & (merged["mutual_info_bits"] < 0.1)
        for row in merged[~weak].itertuples():
            ax.annotate(row.feature.replace("Horizontal_Distance_To_", "HD_")
                        .replace("Vertical_Distance_To_", "VD_"),
                        (row.mutual_info_bits, row.importance_mean),
                        textcoords="offset points", xytext=(5, 3), fontsize=7, color=MUTED)
        if weak.any():
            names = ", ".join(sorted(merged.loc[weak, "feature"].str.replace("_", " ")))
            ax.annotate(f"weak both ways:\n{names}",
                        (merged.loc[weak, "mutual_info_bits"].max(),
                         merged.loc[weak, "importance_mean"].max()),
                        textcoords="offset points", xytext=(70, 2), fontsize=7, color=MUTED, va="center",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
        ax.set_xlabel("Mutual information, bits (model-free)")
        ax.set_ylabel("Permutation importance (drop in F1-macro)")
        ax.set_title("Weak alone but important in the model (upper left) = interactions")
    fig.tight_layout()
    return _save(fig, "19_feature_relevance.png")


def run_relevance(df, save=True, mi_sample=MI_SAMPLE):
    """Compute the table, compare it with the model and store CSV/JSON/PNG."""
    table = relevance_table(df, mi_sample=mi_sample)
    importance_csv = config.REPORTS_DIR / "permutation_importance.csv"
    comparison, merged = None, None
    if importance_csv.exists():
        comparison, merged = compare_with_permutation_importance(
            table, pd.read_csv(importance_csv))
    summary = {
        "rows": int(len(df)),
        "mi_sample_rows": int(min(mi_sample, len(df))),
        "class_entropy_bits": round(class_entropy_bits(df[config.TARGET]), 4),
        "ranking_by_mutual_info": table["feature"].tolist(),
        "vs_permutation_importance": comparison,
        "table": table.round(6).to_dict(orient="records"),
    }
    if save:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(config.REPORTS_DIR / RELEVANCE_CSV.name, index=False)
        (config.REPORTS_DIR / RELEVANCE_JSON.name).write_text(json.dumps(summary, indent=2))
        summary["figure"] = str(plot_relevance(table, merged))
    return table, summary


def print_relevance(table, summary):
    """Console view used by the menu."""
    print(f"Class entropy: {summary['class_entropy_bits']:.3f} bits")
    print(f"{'feature':<36} {'test':<15} {'effect':>8} {'MI, bits':>9} {'% H(y)':>7}")
    for row in table.itertuples():
        print(f"{row.feature:<36} {row.test:<15} {row.effect_size:>8.3f} "
              f"{row.mutual_info_bits:>9.3f} {100 * row.mi_share_of_class_entropy:>6.1f}%")
    comparison = summary.get("vs_permutation_importance")
    if comparison:
        print(f"Spearman rho (MI vs permutation importance): {comparison['spearman_rho']}")
