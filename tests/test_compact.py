"""Size / quality trade-off: staged scoring, choice rule, Pareto front, full run."""

import json

import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src import compact, features, models


@pytest.fixture(scope="module")
def small(sample):
    X_train, X_test, y_train, y_test = features.split_data(sample)
    pipeline, _ = models.fit_final("lightgbm", compact.candidate_params(7, rounds=40),
                                   X_train, y_train)
    return pipeline, X_test, y_test


def test_candidate_params_keep_the_selected_weighting():
    params = compact.candidate_params(63)
    assert params["num_leaves"] == 63 and params["class_weight"] == "balanced"
    assert params["n_estimators"] == compact.MAX_ROUNDS


def test_staged_scores_grow_in_size_and_stop_at_the_last_round(small):
    pipeline, X_test, y_test = small
    rows = compact.staged_scores(pipeline, X_test, y_test, checkpoints=(10, 20, 40, 80))
    assert [r["rounds"] for r in rows] == [10, 20, 40]          # 80 > fitted rounds
    sizes = [r["size_mb"] for r in rows]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    assert all(0 <= r["f1_macro"] <= 1 for r in rows)


def test_last_checkpoint_equals_the_full_model(small):
    pipeline, X_test, y_test = small
    last = compact.staged_scores(pipeline, X_test, y_test, checkpoints=(40,))[0]
    assert last["f1_macro"] == pytest.approx(
        round(f1_score(y_test, pipeline.predict(X_test), average="macro"), 4))


def _curve():
    return pd.DataFrame({
        "num_leaves": [31, 31, 63, 63, 255],
        "rounds": [100, 500, 100, 500, 500],
        "f1_macro": [0.900, 0.9545, 0.940, 0.9560, 0.9562],
        "size_mb": [1.0, 5.0, 2.0, 10.0, 95.0],
    })


def test_choose_compact_takes_the_smallest_within_tolerance():
    chosen = compact.choose_compact(_curve(), tolerance=0.002)
    assert (chosen["num_leaves"], chosen["rounds"]) == (31, 500)
    loose = compact.choose_compact(_curve(), tolerance=0.02)
    assert (loose["num_leaves"], loose["rounds"]) == (63, 100)
    strict = compact.choose_compact(_curve(), tolerance=0.0)
    assert strict["num_leaves"] == 255


def test_divergence_flag_marks_the_whole_leaf_count():
    curve = pd.DataFrame({
        "num_leaves": [31, 31, 31, 63, 63],
        "rounds": [100, 200, 500, 100, 200],
        "f1_macro": [0.80, 0.85, 0.72, 0.90, 0.895],     # 31 falls 0.13, 63 only 0.005
        "size_mb": [1, 2, 5, 2, 4],
    })
    flagged = compact.flag_divergence(curve)
    assert flagged.loc[flagged["num_leaves"] == 31, "diverged"].all()
    assert not flagged.loc[flagged["num_leaves"] == 63, "diverged"].any()


def test_diverged_configurations_are_never_chosen():
    curve = _curve().assign(diverged=[True, True, False, False, False])
    chosen = compact.choose_compact(curve, tolerance=0.002)
    assert chosen["num_leaves"] != 31
    with pytest.raises(ValueError, match="diverged"):
        compact.choose_compact(curve.assign(diverged=True))


def test_inner_split_is_stratified_and_disjoint(sample):
    X = sample[features.INPUT_COLS]
    y = sample["Cover_Type"]
    X_inner, X_val, y_inner, y_val = compact.inner_split(X, y)
    assert X_inner.index.intersection(X_val.index).empty
    assert len(X_val) == pytest.approx(compact.VALIDATION_SIZE * len(X), abs=1)


def test_pareto_front_drops_dominated_points():
    curve = pd.concat([_curve(), pd.DataFrame({"num_leaves": [127], "rounds": [300],
                                               "f1_macro": [0.93], "size_mb": [20.0]})])
    front = compact.pareto_front(curve)
    assert 0.93 not in front["f1_macro"].tolist()              # bigger and worse than 63x500
    assert front["size_mb"].is_monotonic_increasing
    assert front["f1_macro"].is_monotonic_increasing


def test_run_compact_on_the_sample(sample, small, tmp_reports, monkeypatch):
    monkeypatch.setattr(models, "load_model", lambda path=None: small[0])
    curve, final, paired = compact.run_compact(sample, leaves=(5, 7), rounds=30,
                                               checkpoints=(10, 30))
    assert set(curve["num_leaves"]) == {5, 7} and len(curve) == 4
    assert "diverged" in curve.columns
    assert set(final) == {"selected", "compact"}
    assert paired["delta_ci_low"] <= paired["delta_f1_macro"] <= paired["delta_ci_high"]
    stored = json.loads((tmp_reports / "compact_final.json").read_text())
    assert stored["chosen_params"]["class_weight"] == "balanced"
    assert stored["chosen_params"]["learning_rate"] == compact.LEARNING_RATE
    assert "pilot_learning_rate_0_1" in stored
    assert final["compact"]["trained_on_rows"] < final["selected"]["trained_on_rows"]
    assert "Compact LightGBM" in stored["verdict"]
    assert (tmp_reports / "compact_candidates.csv").exists()
    assert (tmp_reports / "figures" / compact.FIGURE_NAME).exists()
