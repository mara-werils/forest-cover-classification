"""Feature ablation: set definitions, column selection and the full run on the sample."""

import json
import math

import pandas as pd
import pytest

from src import ablation, config, features


def test_reference_and_raw_sets():
    sets = ablation.feature_sets()
    assert sets[ablation.REFERENCE_SET] == features.NUMERIC_MODEL_COLS
    assert sets["raw_only"] == config.NUMERIC_COLS
    assert list(sets)[0] == ablation.REFERENCE_SET


def test_groups_partition_the_engineered_columns():
    grouped = [c for cols in ablation.FEATURE_GROUPS.values() for c in cols]
    assert len(grouped) == len(set(grouped)), "a column is in two groups"
    assert set(grouped) == set(features.ENGINEERED_COLS)


@pytest.mark.parametrize("group", list(ablation.FEATURE_GROUPS))
def test_each_leave_one_out_set_drops_exactly_its_group(group):
    kept = ablation.feature_sets()[f"without_{group}"]
    removed = set(features.NUMERIC_MODEL_COLS) - set(kept)
    assert removed == set(ablation.FEATURE_GROUPS[group])
    # Order of the remaining columns is unchanged.
    assert kept == [c for c in features.NUMERIC_MODEL_COLS if c in kept]


def test_default_preprocessors_are_unchanged(sample):
    X = sample[features.INPUT_COLS]
    out = features.tree_preprocessor().fit_transform(X)
    assert out.shape[1] == len(features.NUMERIC_MODEL_COLS) + 2


@pytest.mark.parametrize("set_name", ["raw_only", "without_distance_combinations"])
def test_preprocessors_use_only_the_selected_columns(sample, set_name):
    X = sample[features.INPUT_COLS]
    columns = ablation.feature_sets()[set_name]
    tree = features.tree_preprocessor(columns).fit_transform(X)
    assert tree.shape[1] == len(columns) + 2
    linear = features.linear_preprocessor(columns).fit_transform(X)
    one_hot = X[config.WILDERNESS].nunique() + X[config.SOIL].nunique()
    assert linear.shape[1] == len(columns) + one_hot


def test_hist_gb_categorical_indices_follow_the_feature_set():
    columns = ablation.feature_sets()["raw_only"]
    pipeline = ablation.build_ablation_pipeline("hist_gb", columns)
    assert pipeline.get_params()["model__categorical_features"] == [10, 11]


def test_tuned_params_are_applied():
    pipeline = ablation.build_ablation_pipeline("logreg", config.NUMERIC_COLS, {"C": 0.1})
    assert pipeline.get_params()["model__C"] == 0.1


def test_deltas_and_verdicts():
    rows = []
    for model, values in {"m1": [0.90, 0.85, 0.90, 0.90, 0.90, 0.80],
                          "m2": [0.70, 0.70, 0.70, 0.70, 0.70, 0.70]}.items():
        for set_name, f1 in zip(ablation.feature_sets(), values):
            rows.append({"model": model, "feature_set": set_name, "f1_macro": f1,
                         "accuracy": f1})
    table = ablation.add_deltas(pd.DataFrame(rows))
    assert (table.loc[table["feature_set"] == ablation.REFERENCE_SET, "delta_f1_macro"]
            == 0).all()
    m1 = table[table["model"] == "m1"].set_index("feature_set")["delta_f1_macro"]
    assert m1["raw_only"] == pytest.approx(-0.10)
    lines = ablation.verdicts(table)
    assert "add +0.1000" in lines[0] and "hydrology_geometry" in lines[0]
    assert "not at all" in lines[1] and "no single group" in lines[1]


def test_run_ablation_on_the_sample(sample, tmp_reports):
    table = ablation.run_ablation(sample, model_names=("logreg", "lightgbm"))
    assert len(table) == 2 * len(ablation.feature_sets())
    assert table["f1_macro"].between(0, 1).all()
    stored = json.loads((tmp_reports / "feature_ablation.json").read_text())
    assert stored["test_rows"] == math.ceil(len(sample) * config.TEST_SIZE)
    assert len(stored["verdicts"]) == 2
    assert (tmp_reports / "feature_ablation.csv").exists()
    assert (tmp_reports / "figures" / ablation.FIGURE_NAME).exists()


def test_train_rows_subsamples_train_only(sample, tmp_reports):
    table = ablation.run_ablation(sample, model_names=("logreg",), train_rows=300, save=False)
    assert len(table) == len(ablation.feature_sets())
    assert not (tmp_reports / "feature_ablation.csv").exists()


def test_missing_tuning_results_is_a_clear_error(sample, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TUNING_JSON", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="step 5"):
        ablation.run_ablation(sample)
