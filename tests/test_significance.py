"""Bootstrap intervals, paired tests and McNemar against known answers."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src import config, significance


def _labels(n, seed):
    return np.random.default_rng(seed).integers(1, 8, n)


def _noisy(y, error_rate, seed):
    """Copy of y where a share of the labels is replaced by a random class."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    flip = rng.random(len(y)) < error_rate
    out[flip] = rng.integers(1, 8, flip.sum())
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_f1_from_confusion_matches_sklearn(seed):
    y = _labels(3_000, seed)
    pred = _noisy(y, 0.3, seed + 10)
    confusion = significance.confusion_counts(y, pred)
    assert confusion.sum() == len(y)
    assert significance.f1_macro_from_confusion(confusion) == pytest.approx(
        f1_score(y, pred, average="macro"))


def test_f1_handles_a_class_that_is_never_predicted():
    y = np.array([1, 1, 2, 2, 3])
    pred = np.array([1, 1, 2, 2, 2])          # class 3 never predicted
    expected = f1_score(y, pred, labels=significance.LABELS, average="macro", zero_division=0)
    assert significance.f1_macro_from_confusion(
        significance.confusion_counts(y, pred)) == pytest.approx(expected)


def test_batch_of_matrices_gives_one_score_each():
    y = _labels(500, 0)
    one = significance.confusion_counts(y, _noisy(y, 0.2, 1))
    batch = np.stack([one, one, one])
    assert significance.f1_macro_from_confusion(batch).shape == (3,)


def test_bootstrap_interval_contains_the_point_and_shrinks_with_n():
    small_y = _labels(1_000, 0)
    large_y = _labels(50_000, 0)
    small = significance.bootstrap_f1(small_y, _noisy(small_y, 0.2, 1), n_boot=500)
    large = significance.bootstrap_f1(large_y, _noisy(large_y, 0.2, 1), n_boot=500)
    for result in (small, large):
        assert result["ci_low"] <= result["f1_macro"] <= result["ci_high"]
    assert large["ci_width"] < small["ci_width"] / 3        # ~ 1 / sqrt(n)


def test_bootstrap_is_reproducible_with_the_same_seed():
    y = _labels(2_000, 0)
    pred = _noisy(y, 0.2, 1)
    assert significance.bootstrap_f1(y, pred, n_boot=200) == significance.bootstrap_f1(
        y, pred, n_boot=200)


def test_paired_bootstrap_identical_models_is_exactly_zero():
    y = _labels(2_000, 0)
    pred = _noisy(y, 0.2, 1)
    result = significance.paired_bootstrap(y, pred, pred, n_boot=300)
    assert result["delta_f1_macro"] == 0
    assert result["delta_ci_low"] == 0 == result["delta_ci_high"]


def test_paired_bootstrap_detects_a_clearly_better_model():
    y = _labels(20_000, 0)
    good, bad = _noisy(y, 0.05, 1), _noisy(y, 0.30, 2)
    result = significance.paired_bootstrap(y, good, bad, n_boot=500)
    assert result["delta_f1_macro"] == pytest.approx(
        f1_score(y, good, average="macro") - f1_score(y, bad, average="macro"))
    assert result["delta_ci_low"] > 0
    assert result["share_a_not_better"] == 0


def test_mcnemar_counts_and_p_value():
    y = np.ones(100, dtype=int)
    a = y.copy()
    b = y.copy()
    a[:10] = 2                      # A wrong on 10 cells where B is right
    b[10:40] = 2                    # B wrong on 30 cells where A is right
    result = significance.mcnemar(y, a, b)
    assert (result["only_a_right"], result["only_b_right"]) == (30, 10)
    assert result["mcnemar_p"] == pytest.approx(0.0022, abs=2e-4)   # exact binomial, 40 trials
    assert significance.mcnemar(y, a, a)["mcnemar_p"] == 1.0


def test_holm_adjustment():
    adjusted = significance.holm([0.01, 0.04, 0.03, 0.20])
    # Sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> monotone 0.09, 0.20*1 -> 0.20
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert significance.holm([0.5, 0.9])[1] == 1.0


def _fake_predictions(n=5_000):
    y = _labels(n, 0)
    return pd.DataFrame({"y_true": y,
                         "baseline_dummy": np.full(n, 2),
                         "random_forest": _noisy(y, 0.10, 1),
                         "hist_gb": _noisy(y, 0.08, 2),
                         "lightgbm": _noisy(y, 0.08, 3)})


def test_pair_table_orders_and_flags_pairs():
    pairs = significance.pair_table(_fake_predictions(), n_boot=300)
    assert len(pairs) == 3                                   # baseline is not paired
    assert (pairs["delta_f1_macro"] >= 0).all()              # better model is always A
    assert pairs["mcnemar_p_holm"].ge(pairs["mcnemar_p"]).all()
    lines = significance.verdicts(pairs)
    assert len(lines) == 2 and all("LightGBM" in line for line in lines)


def test_run_significance_writes_outputs(tmp_reports):
    models_, pairs = significance.run_significance(predictions=_fake_predictions(), n_boot=200)
    assert models_.iloc[-1]["model"] == "baseline_dummy"
    stored = json.loads((tmp_reports / "significance.json").read_text())
    assert stored["n_bootstrap"] == 200 and len(stored["pairs"]) == len(pairs)
    for name in ("significance_models.csv", "significance_pairs.csv",
                 "test_predictions.parquet", f"figures/{significance.FIGURE_NAME}"):
        assert (tmp_reports / name).exists()


def test_collect_predictions_on_the_sample(sample):
    frame = significance.collect_predictions(sample, model_names=("baseline_dummy", "logreg"))
    assert list(frame.columns) == ["y_true", "baseline_dummy", "logreg"]
    assert frame.notna().all().all()
    assert len(frame) == int(np.ceil(len(sample) * config.TEST_SIZE))
