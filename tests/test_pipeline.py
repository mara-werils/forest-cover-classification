"""Fast tests on the 1,000-row sample committed to the repository.

Run with:  python -m pytest -q
"""

import pandas as pd
import pytest

from src import audit, config, data_loader, database, features, predict
from src.queries import QUERIES


@pytest.fixture(scope="module")
def sample():
    df = pd.read_csv(config.SAMPLE_CSV, dtype=data_loader.optimized_dtypes())
    return data_loader.collapse_one_hot(df)


def test_sample_has_all_classes(sample):
    assert set(sample[config.TARGET]) == set(config.COVER_TYPES)


def test_one_hot_collapse_is_consistent(sample):
    for _, row in sample.head(50).iterrows():
        assert row[f"Wilderness_Area{row[config.WILDERNESS]}"] == 1
        assert row[f"Soil_Type{row[config.SOIL]}"] == 1


def test_engineered_features_have_no_nans(sample):
    out = features.add_engineered_features(sample[features.INPUT_COLS])
    assert not out[features.ENGINEERED_COLS].isna().any().any()


def test_audit_finds_no_integrity_problems(sample):
    report = audit.run_audit(sample, save=False)
    assert report["one_hot_integrity"]["soil"]["rows_with_several"] == 0
    assert sum(report["range_violations"].values()) == 0


def test_validation_rejects_bad_input():
    record = dict(predict.EXAMPLE_RECORD)
    record["Slope"] = 120
    with pytest.raises(ValueError):
        predict.validate_record(record)
    record = dict(predict.EXAMPLE_RECORD)
    record["Elevation"] = "high"
    with pytest.raises(ValueError):
        predict.validate_record(record)


def test_database_and_descriptive_queries(sample, tmp_path):
    db_path = tmp_path / "test.db"
    counts = database.build_database(sample, db_path=db_path)
    assert counts["cell"] == len(sample)
    assert counts["soil_type"] == 40
    for key, query in QUERIES.items():
        if query.get("requires_predictions"):
            continue
        frame = database.read_sql(query["sql"], db_path=db_path)
        assert isinstance(frame, pd.DataFrame)
    shares = database.read_sql(QUERIES["Q01_class_distribution"]["sql"], db_path=db_path)
    assert abs(shares["share_pct"].sum() - 100) < 0.1


def test_area_scores_flag_unseen_classes():
    from src.robustness import _area_scores

    scores = _area_scores([1, 1, 4, 4], [1, 1, 1, 1], seen={1, 2})
    assert scores["classes_unseen_in_train"] == [4]
    assert scores["accuracy"] == 0.5
