"""Streamlit dashboard: filters, interactive EDA, SQL, model results, prediction.

Run with:  streamlit run app.py
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, data_loader, models, predict, queries
from src.eda import CLASS_COLORS

st.set_page_config(page_title="Forest Cover Type", layout="wide")
COLOR_MAP = {config.COVER_TYPES[k]: v for k, v in CLASS_COLORS.items()}
CLASS_ORDER = list(config.COVER_TYPES.values())


@st.cache_data(show_spinner="Loading Parquet ...")
def load_data():
    df = data_loader.load_parquet(columns=config.NUMERIC_COLS + [config.WILDERNESS, config.SOIL,
                                                                  config.TARGET])
    df["Cover type"] = df[config.TARGET].map(config.COVER_TYPES)
    df["Wilderness area"] = df[config.WILDERNESS].map(config.WILDERNESS_AREAS)
    return df


@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    return models.load_model()


def sidebar_filters(df):
    """Filters shared by all tabs: area, cover type, elevation, soil."""
    st.sidebar.header("Filters")
    areas = st.sidebar.multiselect("Wilderness area", list(config.WILDERNESS_AREAS.values()),
                                   default=list(config.WILDERNESS_AREAS.values()))
    classes = st.sidebar.multiselect("Cover type", CLASS_ORDER, default=CLASS_ORDER)
    low, high = int(df["Elevation"].min()), int(df["Elevation"].max())
    elevation = st.sidebar.slider("Elevation, m", low, high, (low, high), step=10)
    soils = st.sidebar.multiselect("Soil type (empty = all)", list(range(1, 41)))
    mask = (
        df["Wilderness area"].isin(areas)
        & df["Cover type"].isin(classes)
        & df["Elevation"].between(*elevation)
    )
    if soils:
        mask &= df[config.SOIL].isin(soils)
    return df[mask]


def tab_overview(view, total):
    cols = st.columns(4)
    cols[0].metric("Cells in selection", f"{len(view):,}", f"{len(view) / total:.1%} of data",
                   delta_color="off")
    cols[1].metric("Mean elevation", f"{view['Elevation'].mean():,.0f} m" if len(view) else "-")
    cols[2].metric("Cover types present", view["Cover type"].nunique())
    cols[3].metric("Soil types present", view[config.SOIL].nunique())
    if view.empty:
        st.warning("No rows match the filters.")
        return
    counts = view["Cover type"].value_counts().reindex(CLASS_ORDER).dropna().reset_index()
    counts.columns = ["Cover type", "Cells"]
    fig = px.bar(counts, x="Cells", y="Cover type", color="Cover type", orientation="h",
                 color_discrete_map=COLOR_MAP, category_orders={"Cover type": CLASS_ORDER},
                 title="Class distribution in the selection")
    fig.update_layout(showlegend=False, template="plotly_white")
    st.plotly_chart(fig, width="stretch")


def tab_explore(view):
    if view.empty:
        st.warning("No rows match the filters.")
        return
    feature = st.selectbox("Feature", config.NUMERIC_COLS, index=0)
    fig = px.box(view.sample(min(len(view), 50_000), random_state=0), x="Cover type", y=feature,
                 color="Cover type", color_discrete_map=COLOR_MAP,
                 category_orders={"Cover type": CLASS_ORDER},
                 title=f"{feature} by cover type (up to 50k sampled cells)")
    fig.update_layout(showlegend=False, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    sample = view.sample(min(len(view), 15_000), random_state=0)
    x_axis = st.selectbox("Scatter X axis", config.NUMERIC_COLS, index=5)
    fig = px.scatter(sample, x=x_axis, y="Elevation", color="Cover type", opacity=0.5,
                     color_discrete_map=COLOR_MAP, category_orders={"Cover type": CLASS_ORDER},
                     hover_data=["Wilderness area", config.SOIL],
                     title=f"Elevation vs {x_axis} ({len(sample):,} sampled cells)")
    fig.update_traces(marker={"size": 4})
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width="stretch")


def tab_sql():
    items = queries.list_queries()
    labels = [f"{key}: {title}" for key, title, _ in items]
    choice = st.selectbox("Query", labels)
    key = items[labels.index(choice)][0]
    st.caption(f"SQL features: {queries.QUERIES[key]['features']}")
    st.code(queries.QUERIES[key]["sql"].strip(), language="sql")
    try:
        frame, seconds = queries.run_query(key, save=False)
        st.dataframe(frame, width="stretch")
        st.caption(f"{len(frame)} rows in {seconds:.3f} s")
    except FileNotFoundError as error:
        st.error(str(error))


def tab_models():
    if not config.RESULTS_PARQUET.exists():
        st.info("Train the models first: python main.py, option 5.")
        return
    results = pd.read_parquet(config.RESULTS_PARQUET)
    show = ["model", "accuracy", "f1_macro", "balanced_accuracy", "train_seconds",
            "predict_ms_per_1000", "model_size_mb", "peak_memory_mb"]
    st.dataframe(results[show].sort_values("f1_macro", ascending=False),
                 width="stretch", hide_index=True)
    for name in ("11_model_comparison.png", "12_per_class_f1.png", "10_confusion_best.png",
                 "13_permutation_importance.png", "14_error_by_elevation.png",
                 "15_learning_curve.png"):
        path = config.FIGURES_DIR / name
        if path.exists():
            st.image(str(path), width="stretch")


def tab_predict():
    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        return
    meta = json.loads(config.BEST_MODEL_META.read_text())
    st.caption(f"Model: {meta['model']} (test F1-macro {meta['metrics']['f1_macro']:.4f})")
    with st.form("predict"):
        cols = st.columns(3)
        record = {}
        for i, (column, (low, high)) in enumerate(predict.INPUT_LIMITS.items()):
            if column == config.WILDERNESS:
                name = cols[i % 3].selectbox("Wilderness area",
                                             list(config.WILDERNESS_AREAS.values()))
                record[column] = [k for k, v in config.WILDERNESS_AREAS.items() if v == name][0]
            else:
                record[column] = cols[i % 3].number_input(
                    column, min_value=low, max_value=high,
                    value=predict.EXAMPLE_RECORD[column], step=1)
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            result = predict.predict_record(record, model=model)
        except ValueError as error:
            st.error(str(error))
            return
        predict.append_history(record, result)
        st.success(f"Most likely: **{result[0]['cover_type']}** ({result[0]['probability']:.1%})")
        st.dataframe(pd.DataFrame(result), hide_index=True)


def main():
    st.title("Forest Cover Type Classification")
    st.caption("UCI Covertype: 581,012 cells of 30x30 m, Roosevelt National Forest, Colorado")
    try:
        df = load_data()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    view = sidebar_filters(df)
    tabs = st.tabs(["Overview", "Explore", "SQL", "Models", "Predict"])
    with tabs[0]:
        tab_overview(view, len(df))
    with tabs[1]:
        tab_explore(view)
    with tabs[2]:
        tab_sql()
    with tabs[3]:
        tab_models()
    with tabs[4]:
        tab_predict()


main()
