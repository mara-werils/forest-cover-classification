"""SHAP analysis: exactness of the decomposition, samples and the summary tables."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config, features, models, shap_analysis

SMALL_LGBM = {"n_estimators": 20, "num_leaves": 15, "learning_rate": 0.1,
              "min_child_samples": 5, "class_weight": None}


@pytest.fixture(scope="module")
def fitted(sample):
    X_train, X_test, y_train, y_test = features.split_data(sample)
    pipeline, _ = models.fit_final("lightgbm", SMALL_LGBM, X_train, y_train)
    return pipeline, X_test, y_test


def test_shap_values_add_up_to_the_raw_score(fitted):
    pipeline, X_test, _ = fitted
    values, bias = shap_analysis.shap_values(pipeline, X_test.iloc[:50])
    model = pipeline.named_steps["model"]
    raw = model.predict(pipeline.named_steps["prep"].transform(X_test.iloc[:50]), raw_score=True)
    assert values.shape == (50, len(model.classes_), len(features.NUMERIC_MODEL_COLS) + 2)
    assert values.sum(axis=2) + bias == pytest.approx(raw, abs=1e-8)


def test_only_lightgbm_is_accepted(sample):
    X_train, _, y_train, _ = features.split_data(sample)
    forest, _ = models.fit_final("random_forest", {"n_estimators": 5}, X_train, y_train)
    with pytest.raises(ValueError, match="LightGBM"):
        shap_analysis.split_pipeline(forest)


def test_balanced_sample_caps_every_class(sample):
    y = sample[config.TARGET]
    index = shap_analysis.balanced_sample(y, per_class=20)
    assert index.is_unique
    counts = y.loc[index].value_counts()
    for label, available in y.value_counts().items():
        assert counts[label] == min(20, available)


def test_pair_sample_groups_have_the_right_labels():
    y = pd.Series([1, 1, 2, 2, 1, 2, 3])
    predictions = pd.Series([1, 2, 2, 1, 1, 2, 3])
    groups = shap_analysis.pair_sample(y, predictions, per_group=10)
    for name, (true, pred) in shap_analysis.PAIR_GROUPS.items():
        assert (y.loc[groups[name]] == true).all()
        assert (predictions.loc[groups[name]] == pred).all()
    assert len(groups["Spruce/Fir, correct"]) == 2


def test_per_class_importance_known_values():
    values = np.zeros((4, 2, 3))
    values[:2, 0, 0] = [1.0, -3.0]          # class 1 cells, column a
    values[2:, 1, 2] = [0.5, 0.5]           # class 2 cells, column c
    table = shap_analysis.per_class_importance(values, [1, 1, 2, 2], [1, 2], ["a", "b", "c"])
    assert table.loc["a", "Spruce/Fir"] == pytest.approx(2.0)
    assert table.loc["c", "Lodgepole"] == pytest.approx(0.5)
    assert table.index[0] == "a"            # sorted by the overall column


def test_group_importance_sums_engineered_columns_before_abs():
    names = features.feature_names_after_preprocessing("tree")
    values = np.zeros((1, 1, len(names)))
    values[0, 0, names.index("Aspect_Sin")] = 2.0
    values[0, 0, names.index("Aspect_Cos")] = -2.0   # cancels out inside the group
    values[0, 0, names.index("Elevation")] = 1.0
    shares = shap_analysis.group_importance(values, names)
    assert shares.sum() == pytest.approx(1)
    assert shares["Elevation"] == pytest.approx(1)
    assert shares["[aspect_sin_cos]"] == 0
    assert "Aspect_Sin" not in shares.index


def test_pair_margin_sign():
    values = np.zeros((1, 3, 2))
    values[0, 0, 0] = 1.0                    # class 1 (Spruce/Fir) gets +1 from column 0
    values[0, 1, 1] = 2.0                    # class 2 (Lodgepole) gets +2 from column 1
    margins = shap_analysis.pair_margin_contributions(values, [1, 2, 3])
    assert margins[0].tolist() == [1.0, -2.0]


def test_crossover_elevation_on_a_known_line():
    elevation = np.arange(2700, 3400, 2)
    contribution = (elevation - 3010) / 100
    assert shap_analysis.crossover_elevation(elevation, contribution) == pytest.approx(3010,
                                                                                       abs=30)
    assert shap_analysis.crossover_elevation(elevation, np.ones_like(elevation)) is None


def test_explain_record_matches_the_prediction(fitted):
    pipeline, X_test, _ = fitted
    record = X_test.iloc[0].to_dict()
    result = shap_analysis.explain_record(record, pipeline, top=4)
    predicted = pipeline.predict(features.to_input_frame(record))[0]
    assert result["cover_type"] == config.COVER_TYPES[int(predicted)]
    sizes = [abs(c["contribution"]) for c in result["top_contributions"]]
    assert len(sizes) == 4 and sizes == sorted(sizes, reverse=True)


def test_run_shap_writes_outputs(sample, fitted, tmp_reports, monkeypatch):
    pipeline = fitted[0]
    monkeypatch.setattr(models, "load_model", lambda path=None: pipeline)
    importance, pair, summary = shap_analysis.run_shap(sample, per_class=5, per_group=3)
    assert "all classes" in importance.columns
    assert summary["example_record"]["cover_type"] in config.COVER_TYPES.values()
    stored = json.loads((tmp_reports / "shap_summary.json").read_text())
    assert stored["cells_explained"] == summary["cells_explained"]
    for name in ("shap_per_class.csv", "shap_groups.csv", "shap_spruce_vs_lodgepole.csv",
                 f"figures/{shap_analysis.FIGURE_NAME}"):
        assert (tmp_reports / name).exists()
