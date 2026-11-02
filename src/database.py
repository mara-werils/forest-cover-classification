"""SQLite storage with a normalised schema.

Tables
------
cover_type(cover_type_id PK, name)
wilderness_area(wilderness_id PK, name)
climatic_zone(climatic_zone_id PK, name)
geologic_zone(geologic_zone_id PK, name)
soil_type(soil_id PK, elu_code, description,
          climatic_zone_id FK, geologic_zone_id FK)
cell(cell_id PK, 10 numeric measurements,
     wilderness_id FK, soil_id FK, cover_type_id FK)
split_assignment(cell_id PK/FK, split)          -- train / test
model(model_id PK, name, family, params_json)
experiment(experiment_id PK, model_id FK, metrics ..., created_at)
prediction(cell_id FK, model_id FK, predicted_cover_type_id FK)

The 44 one-hot columns of the raw file are replaced by two foreign keys,
so the fact table `cell` has 14 columns instead of 55.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

from src import config

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cover_type (
    cover_type_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS wilderness_area (
    wilderness_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS climatic_zone (
    climatic_zone_id INTEGER PRIMARY KEY,
    name             TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS geologic_zone (
    geologic_zone_id INTEGER PRIMARY KEY,
    name             TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS soil_type (
    soil_id          INTEGER PRIMARY KEY,
    elu_code         INTEGER NOT NULL,
    description      TEXT NOT NULL,
    climatic_zone_id INTEGER NOT NULL REFERENCES climatic_zone(climatic_zone_id),
    geologic_zone_id INTEGER NOT NULL REFERENCES geologic_zone(geologic_zone_id)
);
CREATE TABLE IF NOT EXISTS cell (
    cell_id                            INTEGER PRIMARY KEY,
    elevation                          INTEGER NOT NULL,
    aspect                             INTEGER NOT NULL,
    slope                              INTEGER NOT NULL,
    horizontal_distance_to_hydrology   INTEGER NOT NULL,
    vertical_distance_to_hydrology     INTEGER NOT NULL,
    horizontal_distance_to_roadways    INTEGER NOT NULL,
    hillshade_9am                      INTEGER NOT NULL,
    hillshade_noon                     INTEGER NOT NULL,
    hillshade_3pm                      INTEGER NOT NULL,
    horizontal_distance_to_fire_points INTEGER NOT NULL,
    wilderness_id  INTEGER NOT NULL REFERENCES wilderness_area(wilderness_id),
    soil_id        INTEGER NOT NULL REFERENCES soil_type(soil_id),
    cover_type_id  INTEGER NOT NULL REFERENCES cover_type(cover_type_id)
);
CREATE TABLE IF NOT EXISTS split_assignment (
    cell_id INTEGER PRIMARY KEY REFERENCES cell(cell_id),
    split   TEXT NOT NULL CHECK (split IN ('train', 'test'))
);
CREATE TABLE IF NOT EXISTS model (
    model_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    family      TEXT NOT NULL,
    params_json TEXT
);
CREATE TABLE IF NOT EXISTS experiment (
    experiment_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id          INTEGER NOT NULL REFERENCES model(model_id),
    accuracy          REAL,
    f1_macro          REAL,
    balanced_accuracy REAL,
    train_seconds     REAL,
    predict_seconds   REAL,
    model_size_mb     REAL,
    peak_memory_mb    REAL,
    created_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction (
    cell_id                 INTEGER NOT NULL REFERENCES cell(cell_id),
    model_id                INTEGER NOT NULL REFERENCES model(model_id),
    predicted_cover_type_id INTEGER NOT NULL REFERENCES cover_type(cover_type_id),
    PRIMARY KEY (cell_id, model_id)
);
CREATE INDEX IF NOT EXISTS idx_cell_cover ON cell(cover_type_id);
CREATE INDEX IF NOT EXISTS idx_cell_soil ON cell(soil_id);
CREATE INDEX IF NOT EXISTS idx_cell_wild ON cell(wilderness_id);
"""

# DataFrame column -> SQL column of the `cell` table.
CELL_COLUMNS = {col: col.lower() for col in config.NUMERIC_COLS}
CELL_COLUMNS.update(
    {config.WILDERNESS: "wilderness_id", config.SOIL: "soil_id", config.TARGET: "cover_type_id"}
)


def connect(db_path=None):
    """Open a connection with foreign keys enabled."""
    connection = sqlite3.connect(db_path or config.DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_schema(connection):
    """Create all tables (idempotent)."""
    connection.executescript(SCHEMA)


def load_dimensions(connection):
    """Fill the small lookup tables from the dictionaries in config."""
    connection.executemany(
        "INSERT OR REPLACE INTO cover_type VALUES (?, ?)", list(config.COVER_TYPES.items())
    )
    connection.executemany(
        "INSERT OR REPLACE INTO wilderness_area VALUES (?, ?)",
        list(config.WILDERNESS_AREAS.items()),
    )
    connection.executemany(
        "INSERT OR REPLACE INTO climatic_zone VALUES (?, ?)", list(config.CLIMATIC_ZONES.items())
    )
    connection.executemany(
        "INSERT OR REPLACE INTO geologic_zone VALUES (?, ?)", list(config.GEOLOGIC_ZONES.items())
    )
    soil_rows = []
    for soil_id, (elu, description) in config.SOIL_TYPES.items():
        # ELU code 7745: first digit = climatic zone 7, second = geologic zone 7.
        climatic, geologic = int(str(elu)[0]), int(str(elu)[1])
        soil_rows.append((soil_id, elu, description, climatic, geologic))
    connection.executemany("INSERT OR REPLACE INTO soil_type VALUES (?, ?, ?, ?, ?)", soil_rows)
    connection.commit()


def load_cells(connection, df, chunksize=config.CHUNK_SIZE):
    """Insert the fact table in chunks; cell_id = row position + 1."""
    connection.execute("DELETE FROM prediction")
    connection.execute("DELETE FROM split_assignment")
    connection.execute("DELETE FROM cell")
    cells = df[list(CELL_COLUMNS)].rename(columns=CELL_COLUMNS)
    cells.insert(0, "cell_id", range(1, len(cells) + 1))
    cells.to_sql("cell", connection, if_exists="append", index=False, chunksize=chunksize)
    connection.commit()
    return len(cells)


def save_split(connection, train_index, test_index):
    """Store which cell belongs to train and which to test.

    The indices are DataFrame positions, the cell_id is position + 1.
    """
    connection.execute("DELETE FROM split_assignment")
    rows = [(int(i) + 1, "train") for i in train_index]
    rows += [(int(i) + 1, "test") for i in test_index]
    connection.executemany("INSERT INTO split_assignment VALUES (?, ?)", rows)
    connection.commit()


def register_model(connection, name, family, params):
    """Insert a model if new and return its id."""
    connection.execute(
        "INSERT OR IGNORE INTO model (name, family, params_json) VALUES (?, ?, ?)",
        (name, family, json.dumps(params, default=str)),
    )
    connection.execute(
        "UPDATE model SET params_json = ? WHERE name = ?", (json.dumps(params, default=str), name)
    )
    row = connection.execute("SELECT model_id FROM model WHERE name = ?", (name,)).fetchone()
    return row[0]


def save_experiment(connection, model_id, metrics):
    """Append one row of experiment results."""
    connection.execute(
        """INSERT INTO experiment (model_id, accuracy, f1_macro, balanced_accuracy,
               train_seconds, predict_seconds, model_size_mb, peak_memory_mb, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model_id,
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["balanced_accuracy"],
            metrics["train_seconds"],
            metrics["predict_seconds"],
            metrics["model_size_mb"],
            metrics.get("peak_memory_mb"),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    connection.commit()


def save_predictions(connection, model_id, test_index, predictions):
    """Store test-set predictions so errors can be analysed with SQL."""
    connection.execute("DELETE FROM prediction WHERE model_id = ?", (model_id,))
    rows = [(int(i) + 1, model_id, int(p)) for i, p in zip(test_index, predictions)]
    connection.executemany("INSERT INTO prediction VALUES (?, ?, ?)", rows)
    connection.commit()


def build_database(df, db_path=None):
    """Create the schema and load dimensions and facts. Returns row counts."""
    with closing(connect(db_path)) as connection:
        create_schema(connection)
        load_dimensions(connection)
        load_cells(connection, df)
        counts = {}
        for table in ("cover_type", "wilderness_area", "climatic_zone", "geologic_zone",
                      "soil_type", "cell"):
            counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def read_sql(query, params=None, db_path=None):
    """Run a query and return a pandas DataFrame."""
    path = db_path or config.DB_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build the database first (menu step 3).")
    with closing(connect(path)) as connection:
        return pd.read_sql_query(query, connection, params=params)
