"""Every EDA figure renders on the sample and lands in the (temporary) figures folder."""

import matplotlib
import pytest

from src import audit, config, eda

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _style():
    eda._style()


@pytest.mark.parametrize("plot", [
    eda.plot_class_distribution,
    eda.plot_numeric_histograms,
    eda.plot_elevation_density,
    eda.plot_aspect_rose,
])
def test_simple_figures(sample, tmp_reports, plot):
    path = plot(sample)
    assert path.parent == config.FIGURES_DIR
    assert path.stat().st_size > 10_000


def test_boxplots_use_the_audit_outlier_shares(sample, tmp_reports):
    report = audit.run_audit(sample, save=False)
    assert eda.plot_boxplots(sample, report).exists()
    assert eda.plot_boxplots(sample).exists()


def test_correlation_matrix_is_symmetric(sample, tmp_reports):
    path, corr = eda.plot_correlation(sample)
    assert path.exists()
    assert corr.shape == (10, 10)
    assert (corr - corr.T).abs().to_numpy().max() < 1e-12


def test_crosstab_drops_small_categories(sample, tmp_reports):
    path = eda.plot_crosstab(sample, config.SOIL, {}, "soil_test.png", "t", min_count=30)
    assert path.name == "soil_test.png" and path.exists()


def test_memory_figure_is_skipped_without_reports(tmp_reports):
    assert eda.plot_memory_and_formats() is None


def test_interactive_scatter_is_standalone_html(sample, tmp_reports):
    path = eda.interactive_scatter(sample, n=500)
    html = path.read_text()
    assert path.suffix == ".html"
    assert "plotly" in html.lower()


def test_hillshade_shadow_check(make_frame, tmp_reports):
    df = make_frame(n=700)
    # Ten afternoon shadows, all on steep east-facing slopes.
    df.loc[:9, "Hillshade_3pm"] = 0
    df.loc[:9, "Aspect"] = 90
    df.loc[:9, "Slope"] = 35
    df.loc[10:, "Hillshade_3pm"] = df.loc[10:, "Hillshade_3pm"].clip(lower=1)
    path, stats = eda.plot_hillshade_shadows(df)
    assert path.exists()
    assert stats["hillshade_3pm_zero"] == 10
    assert stats["east_facing_steep_share"] == 1.0
    assert stats["median_slope_zero"] == 35
