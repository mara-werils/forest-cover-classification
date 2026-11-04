"""Schema constraints, experiment tracking and all 12 SQL queries on the sample."""

import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from src import config, database, features, queries


@pytest.fixture
def sample_db(sample, tmp_path, monkeypatch):
    """Sample loaded into a temp database that run_query() will use."""
    db_path = tmp_path / "sample.db"
    database.build_database(sample, db_path=db_path)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "SQL_RESULTS_DIR", tmp_path / "sql")
    return db_path


@pytest.fixture
def tracked_db(sample, sample_db):
    """Two fake models with stored test predictions, as after menu step 5."""
    _, X_test, _, y_test = features.split_data(sample)
    train_index = sample.index.difference(X_test.index)
    rng = np.random.default_rng(0)
    perfect = y_test.to_numpy()
    noisy = np.where(rng.random(len(y_test)) < 0.3, 2, perfect)
    with closing(database.connect(sample_db)) as connection:
        database.save_split(connection, train_index, X_test.index)
        for name, predictions, f1 in (("oracle", perfect, 1.0), ("noisy", noisy, 0.6)):
            model_id = database.register_model(connection, name, "test", {"seed": 0})
            database.save_experiment(connection, model_id, {
                "accuracy": f1, "f1_macro": f1, "balanced_accuracy": f1,
                "train_seconds": 0.1, "predict_seconds": 0.01, "model_size_mb": 0.0,
            })
            database.save_predictions(connection, model_id, X_test.index, predictions)
    return sample_db, y_test, noisy


def test_dimension_tables_are_complete(sample_db):
    counts = database.read_sql(
        """SELECT (SELECT COUNT(*) FROM cover_type) AS cover,
                  (SELECT COUNT(*) FROM wilderness_area) AS areas,
                  (SELECT COUNT(*) FROM soil_type) AS soils""", db_path=sample_db)
    assert counts.iloc[0].tolist() == [7, 4, 40]


def test_soil_zones_are_decoded_from_the_elu_code(sample_db):
    soils = database.read_sql("SELECT * FROM soil_type", db_path=sample_db)
    for row in soils.itertuples():
        assert str(row.elu_code)[0] == str(row.climatic_zone_id)
        assert str(row.elu_code)[1] == str(row.geologic_zone_id)


def test_foreign_keys_are_enforced(sample_db):
    with closing(database.connect(sample_db)) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO cell VALUES (999999, 3000, 0, 0, 0, 0, 0, 200, 200, 100, 0, 1, 1, 9)"
        )


def test_split_check_constraint(sample_db):
    with closing(database.connect(sample_db)) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute("INSERT INTO split_assignment VALUES (1, 'validation')")


def test_reloading_cells_replaces_instead_of_appending(sample, sample_db):
    database.build_database(sample, db_path=sample_db)
    n = database.read_sql("SELECT COUNT(*) AS n FROM cell", db_path=sample_db)["n"].item()
    assert n == len(sample)


def test_register_model_is_idempotent(sample_db):
    with closing(database.connect(sample_db)) as connection:
        first = database.register_model(connection, "m", "f", {"a": 1})
        second = database.register_model(connection, "m", "f", {"a": 2})
        params = connection.execute("SELECT params_json FROM model WHERE name='m'").fetchone()
    assert first == second
    assert params[0] == '{"a": 2}'


def test_sql_class_counts_equal_pandas(sample, sample_db):
    sql = queries.run_query("Q01_class_distribution", save=False)[0]
    by_sql = sql.set_index("cover_type_id")["n_cells"].sort_index()
    by_pandas = sample[config.TARGET].value_counts().sort_index()
    assert by_sql.to_numpy().tolist() == by_pandas.to_numpy().tolist()


def test_query_registry_is_well_formed():
    assert len(queries.QUERIES) == 12
    for key, query in queries.QUERIES.items():
        assert key[:3] in {f"Q{i:02d}" for i in range(1, 13)}
        assert query["title"] and query["features"]
        assert query["sql"].strip().upper().startswith(("SELECT", "WITH"))
    assert [k for k, *_ in queries.list_queries()] == list(queries.QUERIES)


def test_unknown_query_key():
    with pytest.raises(KeyError, match="Unknown query"):
        queries.run_query("Q99_nothing")


def test_every_query_runs_and_saves_csv(tracked_db):
    summary = queries.run_all(include_predictions=True)
    assert len(summary) == 12
    # HAVING thresholds are tuned for 581k rows, so some queries are empty on
    # 1,000 rows; they must still run and write a CSV with the right header.
    for key in queries.QUERIES:
        saved = pd.read_csv(config.SQL_RESULTS_DIR / f"{key}.csv")
        assert len(saved.columns) >= 2
    rows = summary.set_index("query")["rows"]
    assert rows["Q01_class_distribution"] == 7
    assert rows["Q10_recall_per_class"] == 2 * 7


def test_recall_query_matches_numpy(tracked_db):
    _, y_test, noisy = tracked_db
    recall = queries.run_query("Q10_recall_per_class", save=False)[0]
    oracle = recall[recall["model"] == "oracle"]
    assert (oracle["recall"] == 1).all()
    noisy_sql = recall[recall["model"] == "noisy"].set_index("cover_type")["recall"]
    for class_id, name in config.COVER_TYPES.items():
        mask = y_test.to_numpy() == class_id
        expected = (noisy[mask] == class_id).mean()
        assert noisy_sql[name] == pytest.approx(expected, abs=1e-4)


def test_best_model_queries_pick_the_top_experiment(tracked_db):
    # The oracle has the best F1-macro, so the "best model" makes no errors.
    confusions = queries.run_query("Q11_top_confusions", save=False)[0]
    assert confusions.empty or (confusions["n"] == 0).all()


def test_pandas_vs_sql_on_the_sample(sample, sample_db, tmp_path, monkeypatch):
    parquet = tmp_path / "sample.parquet"
    sample.to_parquet(parquet, index=False)
    monkeypatch.setattr(config, "PARQUET_PATH", parquet)
    result = queries.pandas_vs_sql()
    assert result["results_identical"] is True
    grouped = sample.groupby([config.WILDERNESS, config.TARGET]).ngroups
    assert result["groups"] == grouped
