"""Calibration metrics and repairs against cases with a known answer."""

import json

import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from src import calibration, config

LABELS = calibration.LABELS


def _calibrated_sample(n=40_000, seed=0):
    """Labels drawn FROM the probabilities, so they are perfectly calibrated."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2.0, (n, len(LABELS)))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(LABELS, p=row) for row in proba])
    return y, proba


@pytest.fixture(scope="module")
def calibrated():
    return _calibrated_sample()


def test_ece_is_near_zero_for_calibrated_probabilities(calibrated):
    y, proba = calibrated
    assert calibration.expected_calibration_error(y, proba) < 0.01


def test_ece_detects_over_confidence(calibrated):
    y, proba = calibrated
    sharpened = calibration.apply_temperature(proba, 0.4)        # T < 1: over-confident
    assert calibration.expected_calibration_error(y, sharpened) > 0.05


def test_reliability_bins_cover_every_cell(calibrated):
    y, proba = calibrated
    table = calibration.reliability_bins(y, proba)
    assert table["cells"].sum() == len(y)
    assert table["mean_confidence"].between(table["bin_low"], table["bin_high"]).all()


def test_perfect_one_hot_prediction():
    y = np.array([1, 2, 3, 7])
    proba = np.eye(len(LABELS))[np.searchsorted(LABELS, y)]
    assert calibration.brier_score(y, proba) == 0
    assert calibration.log_loss(y, proba) == pytest.approx(0, abs=1e-9)
    assert calibration.expected_calibration_error(y, proba) == 0


def test_brier_and_log_loss_known_values(calibrated):
    y = np.array([1, 2])
    uniform = np.full((2, len(LABELS)), 1 / len(LABELS))
    # (1 - 1/7)^2 + 6 * (1/7)^2 = 6/7
    assert calibration.brier_score(y, uniform) == pytest.approx(6 / 7)
    y_big, proba = calibrated
    assert calibration.log_loss(y_big, proba) == pytest.approx(
        sk_log_loss(y_big, proba, labels=LABELS))


def test_temperature_one_is_identity_and_keeps_the_argmax(calibrated):
    _, proba = calibrated
    assert calibration.apply_temperature(proba, 1.0) == pytest.approx(proba)
    for t in (0.3, 3.0):
        scaled = calibration.apply_temperature(proba, t)
        assert scaled.sum(axis=1) == pytest.approx(1)
        assert (scaled.argmax(axis=1) == proba.argmax(axis=1)).all()


@pytest.mark.parametrize("distortion", [0.5, 2.0])
def test_fit_temperature_undoes_a_known_distortion(calibrated, distortion):
    y, proba = calibrated
    distorted = calibration.apply_temperature(proba, distortion)
    assert calibration.fit_temperature(y, distorted) == pytest.approx(1 / distortion, rel=0.1)


def test_cross_fitting_uses_the_other_half(calibrated):
    y, proba = calibrated
    distorted = calibration.apply_temperature(proba, 0.5)
    repaired, temperatures = calibration.cross_fit_temperature(y, distorted)
    assert len(temperatures) == 2 and all(1.6 < t < 2.4 for t in temperatures)
    assert calibration.expected_calibration_error(y, repaired) < 0.015


def test_prior_correction_undoes_balanced_weights():
    """A balanced model sees p(k|x) / n_k; multiplying by n_k restores p(k|x)."""
    counts = np.array([500, 300, 100, 50, 30, 15, 5])
    true = np.full((1, len(LABELS)), 1 / len(LABELS))
    true = true * counts / (true * counts).sum()            # posterior with priors
    balanced = (true / counts) / (true / counts).sum()      # what weighting learns
    assert calibration.prior_correction(balanced, counts) == pytest.approx(true)


def test_risk_coverage_monotone(calibrated):
    y, proba = calibrated
    table = calibration.risk_coverage(y, proba)
    assert table["coverage"].is_monotonic_decreasing
    assert table["accuracy_on_kept"].is_monotonic_increasing
    assert table["share_of_errors_deferred"].between(0, 1).all()


def test_variants_depend_on_class_weight(calibrated):
    y, proba = calibrated
    counts = np.arange(1, 8) * 100
    assert set(calibration.variants(y, proba, counts, None)) == {"raw", "temperature"}
    assert len(calibration.variants(y, proba, counts, "balanced")) == 4


def test_run_calibration_writes_outputs(calibrated, tmp_reports):
    y, proba = calibrated
    counts = np.arange(1, 8) * 100
    over = calibration.apply_temperature(proba, 0.5)
    collected = (y, {"lightgbm": over, "random_forest": proba},
                 counts, {"lightgbm": "balanced", "random_forest": None})
    table, coverage = calibration.run_calibration(collected=collected)
    assert len(table) == 4 + 2
    stored = json.loads((tmp_reports / "calibration.json").read_text())
    assert stored["selected_model_method"] in {"temperature", "prior_corrected+temperature"}
    assert "over-confident" in stored["verdicts"][0]
    assert len(coverage) == len(calibration.THRESHOLDS)
    assert (tmp_reports / "figures" / calibration.FIGURE_NAME).exists()


def test_collect_probabilities_on_the_sample(sample):
    y, probabilities, counts, weights = calibration.collect_probabilities(
        sample, model_names=("logreg",))
    assert probabilities["logreg"].shape == (len(y), len(LABELS))
    assert probabilities["logreg"].sum(axis=1) == pytest.approx(1)
    assert counts.sum() == len(sample) - len(y)
    assert weights == {"logreg": None}
    assert len(y) == int(np.ceil(len(sample) * config.TEST_SIZE))
