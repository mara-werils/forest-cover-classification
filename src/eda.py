"""Exploratory data analysis: figures saved to reports/figures.

Colour rules used everywhere:
* the 7 cover types always keep the same colour (CLASS_COLORS);
* magnitudes (shares, counts) use one blue ramp, light -> dark;
* correlations use a blue <-> red diverging map with a grey midpoint.
"""

import json

import matplotlib

matplotlib.use("Agg")  # no GUI needed, works in console and on servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap

from src import config

# Fixed categorical order: class 1 -> slot 1, ..., class 7 -> slot 7.
CLASS_COLORS = {
    1: "#2a78d6",  # blue    Spruce/Fir
    2: "#eb6834",  # orange  Lodgepole Pine
    3: "#1baf7a",  # aqua    Ponderosa Pine
    4: "#eda100",  # yellow  Cottonwood/Willow
    5: "#e87ba4",  # magenta Aspen
    6: "#008300",  # green   Douglas-fir
    7: "#6250d6",  # violet  Krummholz
}
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a2a1", "#c43d3c"]
)
TEXT = "#2b2b29"
MUTED = "#6b6a63"
GRID = "#e4e3de"


def _style():
    """Shared matplotlib look: thin recessive axes, light grid."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
        }
    )


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def class_names(index):
    """Map cover type ids to names, keeping order."""
    return [config.COVER_TYPES[int(i)] for i in index]


def plot_class_distribution(df):
    """Bar chart of the target with counts and shares (class imbalance)."""
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        class_names(counts.index),
        counts.values,
        color=[CLASS_COLORS[i] for i in counts.index],
        height=0.7,
    )
    ax.invert_yaxis()
    for bar, value in zip(bars, counts.values):
        share = 100 * value / counts.sum()
        ax.text(bar.get_width() + 3000, bar.get_y() + bar.get_height() / 2,
                f"{value:,}  ({share:.2f}%)", va="center", color=TEXT, fontsize=9)
    ax.set_xlabel("Number of 30x30 m cells")
    ax.set_title("Target distribution: Lodgepole Pine is 103x more frequent than Cottonwood/Willow")
    ax.set_xlim(0, counts.max() * 1.3)
    ax.grid(axis="y", visible=False)
    return _save(fig, "01_class_distribution.png")


def plot_numeric_histograms(df):
    """Histogram of every numeric feature (shape, skew, odd spikes)."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, col in zip(axes.ravel(), config.NUMERIC_COLS):
        ax.hist(df[col], bins=60, color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.set_title(col.replace("_", " "), fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Distributions of the 10 numeric features", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_numeric_histograms.png")


def plot_boxplots(df, audit_report=None):
    """IQR boxplots of numeric features by cover type (outlier view)."""
    cols = ["Elevation", "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points", "Vertical_Distance_To_Hydrology",
            "Slope", "Hillshade_9am"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    classes = sorted(config.COVER_TYPES)
    for ax, col in zip(axes.ravel(), cols):
        data = [df.loc[df[config.TARGET] == c, col].to_numpy() for c in classes]
        box = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                         flierprops={"markersize": 1.5, "alpha": 0.3, "markeredgecolor": MUTED},
                         medianprops={"color": TEXT})
        for patch, c in zip(box["boxes"], classes):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.85)
            patch.set_edgecolor("white")
        ax.set_xticks(range(1, 8), [config.COVER_TYPES[c].split("/")[0][:10] for c in classes],
                      rotation=35, fontsize=8)
        title = col.replace("_", " ")
        if audit_report:
            pct = audit_report["iqr_outliers"][col]["outliers_pct"]
            title += f"\n(global IQR outliers: {pct:.2f}%)"
        ax.set_title(title, fontsize=10)
    fig.suptitle("Boxplots by cover type (Tukey 1.5 x IQR)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_boxplots_by_class.png")


def plot_correlation(df):
    """Pearson correlation of the numeric features."""
    corr = df[config.NUMERIC_COLS].corr()
    labels = [c.replace("Horizontal_Distance_To_", "HD_").replace(
        "Vertical_Distance_To_", "VD_") for c in corr.columns]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.6 else TEXT)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Correlation of numeric features")
    fig.tight_layout()
    return _save(fig, "04_correlation_heatmap.png"), corr


def plot_elevation_density(df):
    """Elevation distribution per class: the single strongest separator."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(1850, 3900, 20)
    for c, color in CLASS_COLORS.items():
        values = df.loc[df[config.TARGET] == c, "Elevation"]
        hist, edges = np.histogram(values, bins=bins, density=True)
        ax.plot(edges[:-1], hist, color=color, linewidth=2, label=config.COVER_TYPES[c])
    ax.set_xlabel("Elevation, m")
    ax.set_ylabel("Density within class")
    ax.set_title("Elevation separates classes into altitude belts, but neighbours overlap")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    return _save(fig, "05_elevation_by_class.png")


def plot_crosstab(df, column, mapping, filename, title, min_count=0):
    """Row-normalised heat map: share of each class inside a category."""
    table = pd.crosstab(df[column], df[config.TARGET])
    table = table[table.sum(axis=1) >= min_count]
    shares = table.div(table.sum(axis=1), axis=0) * 100
    height = max(3.5, 0.28 * len(shares) + 1.5)
    fig, ax = plt.subplots(figsize=(9, height))
    image = ax.imshow(shares, cmap=BLUE_RAMP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(7), class_names(shares.columns), rotation=30, ha="right", fontsize=8)
    ylabels = [f"{mapping.get(int(i), i)} (n={table.loc[i].sum():,})" for i in shares.index]
    ax.set_yticks(range(len(shares)), ylabels, fontsize=8)
    ax.grid(False)
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            value = shares.iloc[i, j]
            if value >= 1:
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 55 else TEXT)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="% of rows in category")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, filename)


def plot_memory_and_formats():
    """Bar charts from the memory report and the CSV/Parquet benchmark."""
    if not (config.MEMORY_JSON.exists() and config.FORMAT_BENCH_JSON.exists()):
        return None
    memory = json.loads(config.MEMORY_JSON.read_text())
    bench = pd.DataFrame(json.loads(config.FORMAT_BENCH_JSON.read_text()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    labels = ["default read\n(int64)", "chunked +\ncompact dtypes"]
    values = [memory["default_read"]["memory_mb"], memory["chunked_typed_read"]["memory_mb"]]
    axes[0].bar(labels, values, color=["#86b6ef", "#1c5cab"], width=0.55)
    axes[0].set_title(f"DataFrame memory, MB (-{memory['memory_reduction_pct']}%)")
    for i, v in enumerate(values):
        axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", color=TEXT)

    names = bench["method"].str.replace("_", "\n")
    for ax, column, title in (
        (axes[1], "load_seconds_median", "Load time, s (median of runs)"),
        (axes[2], "peak_ram_mb", "Peak RAM during load, MB"),
    ):
        ax.bar(names, bench[column], color="#2a78d6", width=0.55)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        for i, v in enumerate(bench[column]):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", color=TEXT, fontsize=8)
    fig.tight_layout()
    return _save(fig, "08_memory_and_formats.png")


def interactive_scatter(df, n=30_000):
    """Plotly scatter (HTML): elevation vs distance to roads, hover for details."""
    sample = df.sample(n, random_state=config.RANDOM_STATE).copy()
    sample["Cover type"] = sample[config.TARGET].map(config.COVER_TYPES)
    sample["Wilderness area"] = sample[config.WILDERNESS].map(config.WILDERNESS_AREAS)
    fig = px.scatter(
        sample,
        x="Horizontal_Distance_To_Roadways",
        y="Elevation",
        color="Cover type",
        color_discrete_map={config.COVER_TYPES[k]: v for k, v in CLASS_COLORS.items()},
        category_orders={"Cover type": list(config.COVER_TYPES.values())},
        hover_data=["Wilderness area", config.SOIL, "Slope", "Horizontal_Distance_To_Hydrology"],
        opacity=0.55,
        title=f"Elevation vs distance to roads ({n:,} random cells). Click legend items to filter.",
        template="plotly_white",
    )
    fig.update_traces(marker={"size": 4})
    path = config.FIGURES_DIR / "09_interactive_scatter.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def run_eda(df, audit_report=None):
    """Produce every figure; returns a dict name -> path and key numbers."""
    config.ensure_dirs()
    _style()
    outputs = {
        "class_distribution": plot_class_distribution(df),
        "histograms": plot_numeric_histograms(df),
        "boxplots": plot_boxplots(df, audit_report),
    }
    outputs["correlation"], corr = plot_correlation(df)
    outputs["elevation"] = plot_elevation_density(df)
    outputs["wilderness"] = plot_crosstab(
        df, config.WILDERNESS, config.WILDERNESS_AREAS, "06_wilderness_vs_class.png",
        "Cover-type mix inside each wilderness area (% of row)")
    soil_names = {k: f"Soil {k} / ELU {v[0]}" for k, v in config.SOIL_TYPES.items()}
    outputs["soil"] = plot_crosstab(
        df, config.SOIL, soil_names, "07_soil_vs_class.png",
        "Cover-type mix per soil type (% of row, soils with >= 500 cells)", min_count=500)
    outputs["formats"] = plot_memory_and_formats()
    outputs["interactive"] = interactive_scatter(df)

    # Strongest correlations, for the report.
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5)
    outputs["top_correlations"] = {f"{a} ~ {b}": round(float(v), 3) for (a, b), v in top.items()}
    return outputs
