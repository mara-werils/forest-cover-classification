"""Headless run of the Streamlit dashboard (streamlit.testing.AppTest)."""

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file("app.py", default_timeout=180).run()


def test_dashboard_renders_without_exceptions(app):
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Overview", "Explore", "Data quality", "SQL", "Models", "Predict"]


def test_data_quality_tab_shows_the_audit(app):
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Rows audited"] == "581,012"
    assert metrics["Missing values"] == "0"
    assert metrics["Broken one-hot rows"] == "0"
