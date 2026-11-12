"""Feature engineering and preprocessing pipelines.

All transformations live inside scikit-learn objects, so they are fitted
on the training split only and re-applied unchanged to the test split
and to user input (no leakage).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

ENGINEERED_COLS = [
    "Euclidean_Distance_To_Hydrology",
    "Water_Elevation",
    "Hillshade_Mean",
    "Hydro_Plus_Fire",
    "Abs_Hydro_Minus_Road",
    "Abs_Fire_Minus_Road",
    "Aspect_Sin",
    "Aspect_Cos",
]
CATEGORICAL_COLS = [config.WILDERNESS, config.SOIL]
INPUT_COLS = config.NUMERIC_COLS + CATEGORICAL_COLS
NUMERIC_MODEL_COLS = config.NUMERIC_COLS + ENGINEERED_COLS


def add_engineered_features(frame):
    """Row-wise features; no statistics are learned, so no leakage.

    * Euclidean distance to water combines the horizontal and vertical
      distances;
    * water elevation = elevation minus the vertical distance to water;
    * sums/differences of the three horizontal distances are a known
      strong signal for this data set (Kaggle forest-cover competition);
    * aspect is an angle, so 359 deg and 1 deg must be close: sin/cos.
    """
    out = frame.copy()
    hydro_h = out["Horizontal_Distance_To_Hydrology"].astype("float32")
    hydro_v = out["Vertical_Distance_To_Hydrology"].astype("float32")
    road = out["Horizontal_Distance_To_Roadways"].astype("float32")
    fire = out["Horizontal_Distance_To_Fire_Points"].astype("float32")
    aspect = np.deg2rad(out["Aspect"].astype("float32"))

    out["Euclidean_Distance_To_Hydrology"] = np.sqrt(hydro_h**2 + hydro_v**2)
    out["Water_Elevation"] = out["Elevation"].astype("float32") - hydro_v
    out["Hillshade_Mean"] = out[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].astype(
        "float32").mean(axis=1)
    out["Hydro_Plus_Fire"] = hydro_h + fire
    out["Abs_Hydro_Minus_Road"] = (hydro_h - road).abs()
    out["Abs_Fire_Minus_Road"] = (fire - road).abs()
    out["Aspect_Sin"] = np.sin(aspect)
    out["Aspect_Cos"] = np.cos(aspect)
    return out


def _feature_step():
    return FunctionTransformer(add_engineered_features, validate=False)


def linear_preprocessor(numeric_cols=None):
    """For Logistic Regression: impute + scale numbers, one-hot categories.

    ``numeric_cols`` selects the numeric inputs (default: all raw and
    engineered columns); the feature ablation passes smaller sets.
    """
    numeric_cols = list(numeric_cols) if numeric_cols is not None else NUMERIC_MODEL_COLS
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columns = ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, CATEGORICAL_COLS)]
    )
    return Pipeline([("features", _feature_step()), ("columns", columns)])


def tree_preprocessor(numeric_cols=None):
    """For tree ensembles: no scaling needed, categories kept as integer codes.

    Trees split on thresholds, so monotone rescaling changes nothing;
    keeping Soil as one integer column (instead of 40 one-hot columns)
    makes Random Forest much faster with the same accuracy.
    ``numeric_cols`` works as in ``linear_preprocessor``.
    """
    numeric_cols = list(numeric_cols) if numeric_cols is not None else NUMERIC_MODEL_COLS
    columns = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), CATEGORICAL_COLS),
        ]
    )
    return Pipeline([("features", _feature_step()), ("columns", columns)])


def feature_names_after_preprocessing(kind):
    """Names of the columns the estimator sees (for importances)."""
    if kind == "tree":
        return NUMERIC_MODEL_COLS + CATEGORICAL_COLS
    names = list(NUMERIC_MODEL_COLS)
    names += [f"Wilderness_{i}" for i in config.WILDERNESS_AREAS]
    names += [f"Soil_{i}" for i in config.SOIL_TYPES]
    return names


def split_data(df, test_size=config.TEST_SIZE, seed=config.RANDOM_STATE):
    """Stratified hold-out split, done BEFORE any fitting.

    The data has no time stamp or group id, so a stratified random
    split is the standard protocol; stratification keeps the 0.47%
    Cottonwood/Willow class present in both parts.
    """
    X = df[INPUT_COLS]
    y = df[config.TARGET].astype("int64")
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def stratified_subsample(X, y, n, seed=config.RANDOM_STATE):
    """Stratified subsample used for hyper-parameter search."""
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X_sub, y_sub


def to_input_frame(values):
    """Turn one user record (dict) into a one-row DataFrame for predict."""
    return pd.DataFrame([{col: values[col] for col in INPUT_COLS}])
