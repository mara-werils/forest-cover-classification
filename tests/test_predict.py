"""Input validation, top-k prediction, history file and the committed model."""

import json

import pytest

from src import config, features, models, predict


@pytest.fixture(scope="module")
def small_model(sample):
    """Fast stand-in for the 40 MB LightGBM: logistic regression on the sample."""
    spec = models.model_specs()["logreg"]
    pipeline = models.build_pipeline(spec)
    return pipeline.fit(sample[features.INPUT_COLS], sample[config.TARGET].astype("int64"))


def test_example_record_is_valid():
    assert predict.validate_record(predict.EXAMPLE_RECORD) == predict.EXAMPLE_RECORD


@pytest.mark.parametrize("column", list(predict.INPUT_LIMITS))
def test_limits_are_inclusive(column):
    low, high = predict.INPUT_LIMITS[column]
    for value in (low, high):
        record = dict(predict.EXAMPLE_RECORD, **{column: value})
        assert predict.validate_record(record)[column] == value


@pytest.mark.parametrize("column", list(predict.INPUT_LIMITS))
def test_one_step_outside_the_limits_is_rejected(column):
    low, high = predict.INPUT_LIMITS[column]
    for value in (low - 1, high + 1):
        record = dict(predict.EXAMPLE_RECORD, **{column: value})
        with pytest.raises(ValueError, match=column):
            predict.validate_record(record)


def test_missing_field_is_named_in_the_error():
    record = dict(predict.EXAMPLE_RECORD)
    del record["Hillshade_Noon"]
    with pytest.raises(ValueError, match="Missing value for Hillshade_Noon"):
        predict.validate_record(record)


@pytest.mark.parametrize("bad", [None, "", "abc", [1]])
def test_non_numeric_values_are_rejected(bad):
    record = dict(predict.EXAMPLE_RECORD, Slope=bad)
    with pytest.raises(ValueError, match="Slope must be an integer"):
        predict.validate_record(record)


def test_numeric_strings_from_forms_are_accepted():
    record = {key: str(value) for key, value in predict.EXAMPLE_RECORD.items()}
    assert predict.validate_record(record) == predict.EXAMPLE_RECORD


def test_top_k_is_sorted_and_named(small_model):
    result = predict.predict_record(predict.EXAMPLE_RECORD, model=small_model, top_k=7)
    probabilities = [item["probability"] for item in result]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) == pytest.approx(1, abs=0.01)
    assert {item["cover_type_id"] for item in result} == set(config.COVER_TYPES)
    for item in result:
        assert item["cover_type"] == config.COVER_TYPES[item["cover_type_id"]]


def test_top_k_limits_the_answer(small_model):
    assert len(predict.predict_record(predict.EXAMPLE_RECORD, model=small_model)) == 3
    assert len(predict.predict_record(predict.EXAMPLE_RECORD, model=small_model, top_k=1)) == 1


def test_invalid_record_never_reaches_the_model(small_model):
    with pytest.raises(ValueError):
        predict.predict_record(dict(predict.EXAMPLE_RECORD, Aspect=400), model=small_model)


def test_history_round_trip(tmp_reports):
    assert predict.load_history() == []
    top = [{"cover_type_id": 2, "cover_type": "Lodgepole Pine", "probability": 0.9}]
    assert predict.append_history(predict.EXAMPLE_RECORD, top) == 1
    assert predict.append_history(predict.EXAMPLE_RECORD, top) == 2
    history = predict.load_history()
    assert len(history) == 2
    assert history[0]["input"] == predict.EXAMPLE_RECORD
    assert history[0]["top"] == top
    assert "T" in history[0]["time"]


def test_broken_history_file_is_treated_as_empty(tmp_reports):
    config.HISTORY_JSON.write_text("{not json")
    assert predict.load_history() == []
    predict.append_history(predict.EXAMPLE_RECORD, [])
    assert len(json.loads(config.HISTORY_JSON.read_text())) == 1


def test_committed_model_matches_the_feature_contract():
    """The versioned joblib must accept today's INPUT_COLS and agree with its JSON."""
    meta = json.loads(config.BEST_MODEL_META.read_text())
    assert meta["input_columns"] == features.INPUT_COLS
    model = models.load_model()
    result = predict.predict_record(predict.EXAMPLE_RECORD, model=model)
    assert result[0]["cover_type"] == "Lodgepole Pine"
    assert result[0]["probability"] > 0.9


def test_missing_model_file_gives_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the models first"):
        models.load_model(tmp_path / "nothing.joblib")
