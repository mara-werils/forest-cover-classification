"""Invariants of the engineered features and the preprocessing pipelines."""

import numpy as np
import pandas as pd
import pytest

from src import config, features


def _row(**overrides):
    base = {col: 0 for col in features.INPUT_COLS}
    base.update({"Elevation": 3000, config.WILDERNESS: 1, config.SOIL: 1})
    base.update(overrides)
    return pd.DataFrame([base])


def test_all_engineered_columns_are_added_and_input_is_untouched(sample):
    raw = sample[features.INPUT_COLS]
    before = raw.copy()
    out = features.add_engineered_features(raw)
    assert list(out.columns) == features.INPUT_COLS + features.ENGINEERED_COLS
    pd.testing.assert_frame_equal(raw, before)


def test_hydrology_geometry():
    out = features.add_engineered_features(
        _row(Horizontal_Distance_To_Hydrology=30, Vertical_Distance_To_Hydrology=40)
    )
    # 3-4-5 triangle scaled by 10.
    assert out.loc[0, "Euclidean_Distance_To_Hydrology"] == pytest.approx(50)
    assert out.loc[0, "Water_Elevation"] == pytest.approx(3000 - 40)


def test_distance_sums_and_differences():
    out = features.add_engineered_features(
        _row(Horizontal_Distance_To_Hydrology=200, Horizontal_Distance_To_Roadways=500,
             Horizontal_Distance_To_Fire_Points=100)
    )
    assert out.loc[0, "Hydro_Plus_Fire"] == pytest.approx(300)
    assert out.loc[0, "Abs_Hydro_Minus_Road"] == pytest.approx(300)
    assert out.loc[0, "Abs_Fire_Minus_Road"] == pytest.approx(400)


def test_hillshade_mean():
    out = features.add_engineered_features(_row(Hillshade_9am=90, Hillshade_Noon=210,
                                                Hillshade_3pm=150))
    assert out.loc[0, "Hillshade_Mean"] == pytest.approx(150)


@pytest.mark.parametrize("a, b", [(0, 360), (1, 359), (179, 181)])
def test_aspect_encoding_is_circular(a, b):
    """Angles that are close on the compass must be close after encoding."""
    rows = pd.concat([_row(Aspect=a), _row(Aspect=b)], ignore_index=True)
    out = features.add_engineered_features(rows)
    vectors = out[["Aspect_Sin", "Aspect_Cos"]].to_numpy()
    chord = np.linalg.norm(vectors[0] - vectors[1])
    gap = min(abs(a - b), 360 - abs(a - b))          # shortest way round the compass
    expected = 2 * np.sin(np.deg2rad(gap) / 2)       # chord length on the unit circle
    assert chord == pytest.approx(expected, abs=1e-5)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)


def test_engineered_features_on_the_sample_are_finite(sample):
    out = features.add_engineered_features(sample[features.INPUT_COLS])
    values = out[features.ENGINEERED_COLS].to_numpy(dtype="float64")
    assert np.isfinite(values).all()
    assert (out["Euclidean_Distance_To_Hydrology"] >= 0).all()


@pytest.mark.parametrize("kind", ["tree", "linear"])
def test_preprocessor_width_matches_feature_names(make_frame, kind):
    df = make_frame(n=280)          # every area and every soil occurs
    pre = features.tree_preprocessor() if kind == "tree" else features.linear_preprocessor()
    out = pre.fit_transform(df[features.INPUT_COLS])
    assert out.shape == (len(df), len(features.feature_names_after_preprocessing(kind)))


@pytest.mark.parametrize("kind", ["tree", "linear"])
def test_transform_of_a_row_does_not_depend_on_its_batch(make_frame, kind):
    """No statistic is computed at predict time: row i alone == row i in a batch."""
    df = make_frame(n=280)
    train, test = df.iloc[:200][features.INPUT_COLS], df.iloc[200:][features.INPUT_COLS]
    pre = features.tree_preprocessor() if kind == "tree" else features.linear_preprocessor()
    pre.fit(train)
    batch = pre.transform(test)
    for i in (0, 17, len(test) - 1):
        single = pre.transform(test.iloc[[i]])
        np.testing.assert_allclose(single[0], batch[i], rtol=1e-6)


def test_linear_scaler_is_fitted_on_train_only(make_frame):
    df = make_frame(n=280)
    train = df.iloc[:200][features.INPUT_COLS]
    test = df.iloc[200:][features.INPUT_COLS].assign(Elevation=3800)   # shifted test set
    pre = features.linear_preprocessor().fit(train)
    elevation = features.NUMERIC_MODEL_COLS.index("Elevation")
    assert abs(pre.transform(train)[:, elevation].mean()) < 1e-6
    # Had the scaler seen the test rows, their mean would also be ~0.
    assert pre.transform(test)[:, elevation].mean() > 1


def test_split_is_stratified_and_disjoint(sample):
    X_train, X_test, y_train, y_test = features.split_data(sample)
    assert len(X_test) == pytest.approx(config.TEST_SIZE * len(sample), abs=1)
    assert not set(X_train.index) & set(X_test.index)
    assert set(y_test) == set(config.COVER_TYPES)
    full = sample[config.TARGET].value_counts(normalize=True)
    part = y_train.value_counts(normalize=True)
    assert (part.reindex(full.index) - full).abs().max() < 0.01


def test_split_is_reproducible(sample):
    first = features.split_data(sample)[1].index
    second = features.split_data(sample)[1].index
    assert first.equals(second)


def test_stratified_subsample(sample):
    X, y = sample[features.INPUT_COLS], sample[config.TARGET]
    X_all, _ = features.stratified_subsample(X, y, n=len(X) + 10)
    assert len(X_all) == len(X)
    X_sub, y_sub = features.stratified_subsample(X, y, n=300)
    assert len(X_sub) == 300
    assert set(y_sub) == set(config.COVER_TYPES)


def test_to_input_frame_orders_columns():
    record = {col: i for i, col in enumerate(reversed(features.INPUT_COLS))}
    record["unused"] = 99
    frame = features.to_input_frame(record)
    assert list(frame.columns) == features.INPUT_COLS
    assert len(frame) == 1
