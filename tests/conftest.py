"""Shared fixtures.

* ``sample``: the 1,000-row stratified sample committed to the repository;
* ``make_frame``: factory for small synthetic frames with a known content,
  so every audit / feature check can be tested against an exact answer;
* ``tmp_reports``: redirects every report and figure path to a temporary
  directory, so tests never overwrite the committed results.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, data_loader


@pytest.fixture(scope="session")
def sample():
    df = pd.read_csv(config.SAMPLE_CSV, dtype=data_loader.optimized_dtypes())
    return data_loader.collapse_one_hot(df)


def _synthetic(n=70, seed=0):
    """Valid rows: every value inside VALID_RANGES, one flag per one-hot block.

    Classes cycle 1..7 so each class has n / 7 rows; wilderness areas and
    soils cycle too, so the frame looks like a (tiny) real extract.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "Elevation": rng.integers(2000, 3600, n),
            "Aspect": rng.integers(0, 360, n),
            "Slope": rng.integers(0, 40, n),
            "Horizontal_Distance_To_Hydrology": rng.integers(0, 1000, n),
            "Vertical_Distance_To_Hydrology": rng.integers(-100, 300, n),
            "Horizontal_Distance_To_Roadways": rng.integers(0, 6000, n),
            "Hillshade_9am": rng.integers(100, 255, n),
            "Hillshade_Noon": rng.integers(150, 255, n),
            "Hillshade_3pm": rng.integers(50, 200, n),
            "Horizontal_Distance_To_Fire_Points": rng.integers(0, 6000, n),
        }
    )
    for i, col in enumerate(config.WILDERNESS_COLS):
        frame[col] = (np.arange(n) % 4 == i).astype("uint8")
    for i, col in enumerate(config.SOIL_COLS):
        frame[col] = (np.arange(n) % 40 == i).astype("uint8")
    frame[config.TARGET] = (np.arange(n) % 7 + 1).astype("uint8")
    return data_loader.collapse_one_hot(frame)


@pytest.fixture
def make_frame():
    return _synthetic


@pytest.fixture
def tmp_reports(tmp_path, monkeypatch):
    """Point REPORTS_DIR, FIGURES_DIR and the JSON outputs at tmp_path."""
    reports = tmp_path / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "FIGURES_DIR", figures)
    monkeypatch.setattr(config, "AUDIT_JSON", reports / "data_audit.json")
    monkeypatch.setattr(config, "HISTORY_JSON", reports / "prediction_history.json")
    return reports
