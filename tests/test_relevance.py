"""Effect sizes against cases with a known answer, and the report on the sample."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config, relevance


def test_cramers_v_is_one_for_a_perfect_association():
    x = np.repeat([1, 2, 3, 4], 250)
    assert relevance.cramers_v(x, x * 10)["effect_size"] == pytest.approx(1, abs=0.01)


def test_cramers_v_is_near_zero_for_independent_columns():
    rng = np.random.default_rng(0)
    x, y = rng.integers(1, 5, 20_000), rng.integers(1, 8, 20_000)
    assert relevance.cramers_v(x, y)["effect_size"] < 0.02


def test_kruskal_effect_size_separates_shifted_groups():
    rng = np.random.default_rng(0)
    labels = np.repeat([1, 2, 3], 1_000)
    same = rng.normal(size=3_000)
    shifted = same + labels * 10            # three non-overlapping groups
    assert relevance.kruskal_effect_size(same, labels)["effect_size"] < 0.01
    # Upper bound for k equal, fully separated groups is 1 - 1/k^2.
    best = relevance.kruskal_effect_size(shifted, labels)["effect_size"]
    assert best == pytest.approx(1 - 1 / 3**2, abs=1e-3)


def test_class_entropy_bits():
    assert relevance.class_entropy_bits([1, 2, 3, 4] * 25) == pytest.approx(2)
    assert relevance.class_entropy_bits([5] * 10) == pytest.approx(0)


def test_mutual_information_of_a_copy_of_the_target(sample):
    df = sample.copy()
    df[config.SOIL] = df[config.TARGET]                # Soil now *is* the class
    mi = relevance.mutual_information(df, [config.SOIL, "Aspect"], n=len(df))
    entropy = relevance.class_entropy_bits(df[config.TARGET])
    assert mi[config.SOIL] == pytest.approx(entropy, rel=0.02)
    assert mi["Aspect"] < mi[config.SOIL] / 4


def test_relevance_table_on_the_sample(sample):
    table = relevance.relevance_table(sample, mi_sample=len(sample))
    assert set(table["feature"]) == set(config.NUMERIC_COLS + relevance.CATEGORICAL)
    assert table["mi_rank"].tolist() == list(range(1, len(table) + 1))
    assert table["mutual_info_bits"].is_monotonic_decreasing
    assert (table["mi_share_of_class_entropy"].between(0, 1)).all()
    # Elevation is the strongest single signal in this data set.
    assert table.loc[table["kind"] == "numeric", "feature"].iloc[0] == "Elevation"


def test_comparison_with_permutation_importance(sample):
    table = relevance.relevance_table(sample, mi_sample=len(sample))
    importance = pd.DataFrame({"feature": table["feature"],
                               "importance_mean": table["mutual_info_bits"] * 2})
    comparison, merged = relevance.compare_with_permutation_importance(table, importance)
    assert comparison["spearman_rho"] == pytest.approx(1)
    assert (merged["rank_gap"] == 0).all()


def test_run_relevance_writes_csv_json_and_figure(sample, tmp_reports):
    table, summary = relevance.run_relevance(sample, mi_sample=len(sample))
    stored = json.loads((tmp_reports / "feature_relevance.json").read_text())
    assert stored["ranking_by_mutual_info"] == table["feature"].tolist()
    assert stored["vs_permutation_importance"] is None   # no importance file in tmp
    assert (tmp_reports / "feature_relevance.csv").exists()
    assert (config.FIGURES_DIR / "19_feature_relevance.png").exists()
