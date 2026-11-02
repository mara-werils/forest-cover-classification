"""Single-record prediction with input validation and a history file."""

import json
from datetime import datetime

from src import config, features, models

# Allowed range for each input (wider than the data, narrower than nonsense).
INPUT_LIMITS = {
    "Elevation": (1500, 4500),
    "Aspect": (0, 360),
    "Slope": (0, 90),
    "Horizontal_Distance_To_Hydrology": (0, 10_000),
    "Vertical_Distance_To_Hydrology": (-1_000, 1_000),
    "Horizontal_Distance_To_Roadways": (0, 10_000),
    "Hillshade_9am": (0, 255),
    "Hillshade_Noon": (0, 255),
    "Hillshade_3pm": (0, 255),
    "Horizontal_Distance_To_Fire_Points": (0, 10_000),
    config.WILDERNESS: (1, 4),
    config.SOIL: (1, 40),
}

# A typical Lodgepole Pine cell from the data set, used as menu defaults.
EXAMPLE_RECORD = {
    "Elevation": 2800,
    "Aspect": 150,
    "Slope": 12,
    "Horizontal_Distance_To_Hydrology": 250,
    "Vertical_Distance_To_Hydrology": 40,
    "Horizontal_Distance_To_Roadways": 2000,
    "Hillshade_9am": 220,
    "Hillshade_Noon": 225,
    "Hillshade_3pm": 140,
    "Horizontal_Distance_To_Fire_Points": 1800,
    config.WILDERNESS: 1,
    config.SOIL: 29,
}


def validate_record(record):
    """Check that all fields are present, numeric and inside INPUT_LIMITS.

    Raises ValueError with a readable message on the first problem.
    """
    clean = {}
    for column, (low, high) in INPUT_LIMITS.items():
        if column not in record:
            raise ValueError(f"Missing value for {column}")
        try:
            value = int(record[column])
        except (TypeError, ValueError):
            raise ValueError(f"{column} must be an integer, got {record[column]!r}") from None
        if not low <= value <= high:
            raise ValueError(f"{column} must be between {low} and {high}, got {value}")
        clean[column] = value
    return clean


def predict_record(record, model=None, top_k=3):
    """Return the top-k classes with probabilities for one record."""
    clean = validate_record(record)
    model = model or models.load_model()
    frame = features.to_input_frame(clean)
    probabilities = model.predict_proba(frame)[0]
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    ranked = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
    return [
        {"cover_type_id": int(c), "cover_type": config.COVER_TYPES[int(c)],
         "probability": round(float(p), 4)}
        for c, p in ranked[:top_k]
    ]


def load_history():
    """Previous predictions (empty list if the file is missing or broken)."""
    try:
        return json.loads(config.HISTORY_JSON.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def append_history(record, result):
    """Save the prediction with a time stamp to prediction_history.json."""
    history = load_history()
    history.append({"time": datetime.now().isoformat(timespec="seconds"),
                    "input": record, "top": result})
    config.HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.HISTORY_JSON.write_text(json.dumps(history, indent=2))
    return len(history)
