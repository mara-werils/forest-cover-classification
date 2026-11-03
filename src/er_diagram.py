"""Draw the ER diagram of the SQLite schema (PNG) from a table spec."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src import config

# table -> (x, y of top-left corner, list of (column, key marker))
TABLES = {
    "cover_type": (0.2, 9.4, [("cover_type_id", "PK"), ("name", "")]),
    "wilderness_area": (0.2, 6.6, [("wilderness_id", "PK"), ("name", "")]),
    "climatic_zone": (0.2, 3.8, [("climatic_zone_id", "PK"), ("name", "")]),
    "geologic_zone": (0.2, 1.6, [("geologic_zone_id", "PK"), ("name", "")]),
    "soil_type": (4.0, 3.6, [("soil_id", "PK"), ("elu_code", ""), ("description", ""),
                             ("climatic_zone_id", "FK"), ("geologic_zone_id", "FK")]),
    "cell": (4.0, 10.2, [("cell_id", "PK"), ("elevation", ""), ("aspect", ""), ("slope", ""),
                         ("horiz./vert. distances (4)", ""), ("hillshade_9am/noon/3pm", ""),
                         ("wilderness_id", "FK"), ("soil_id", "FK"), ("cover_type_id", "FK")]),
    "split_assignment": (8.0, 10.2, [("cell_id", "PK,FK"), ("split", "")]),
    "prediction": (8.0, 7.6, [("cell_id", "PK,FK"), ("model_id", "PK,FK"),
                              ("predicted_cover_type_id", "FK")]),
    "model": (8.0, 4.4, [("model_id", "PK"), ("name", ""), ("family", ""), ("params_json", "")]),
    "experiment": (11.6, 4.4, [("experiment_id", "PK"), ("model_id", "FK"), ("accuracy", ""),
                               ("f1_macro", ""), ("balanced_accuracy", ""), ("train_seconds", ""),
                               ("predict_seconds", ""), ("model_size_mb", ""),
                               ("peak_memory_mb", ""), ("created_at", "")]),
}
# (child table, parent table, label)
RELATIONS = [
    ("cell", "cover_type", "N:1"),
    ("cell", "wilderness_area", "N:1"),
    ("cell", "soil_type", "N:1"),
    ("soil_type", "climatic_zone", "N:1"),
    ("soil_type", "geologic_zone", "N:1"),
    ("split_assignment", "cell", "1:1"),
    ("prediction", "cell", "N:1"),
    ("prediction", "model", "N:1"),
    ("experiment", "model", "N:1"),
]
WIDTH = 3.2
ROW = 0.36
STYLE = {"header": "#1c5cab", "pk": "#c43d3c", "line": "#6b6a63", "text": "#2b2b29"}


def _box(ax, name, x, y, columns):
    height = ROW * (len(columns) + 1) + 0.1
    ax.add_patch(FancyBboxPatch((x, y - height), WIDTH, height,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=STYLE["header"], linewidth=1.4))
    ax.add_patch(FancyBboxPatch((x, y - ROW - 0.05), WIDTH, ROW + 0.05,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=STYLE["header"], edgecolor=STYLE["header"]))
    ax.text(x + WIDTH / 2, y - ROW / 2 - 0.02, name, ha="center", va="center", color="white",
            fontweight="bold", fontsize=9.5)
    for i, (column, key) in enumerate(columns):
        cy = y - ROW * (i + 1.5) - 0.05
        ax.text(x + 0.12, cy, column, va="center", fontsize=8, color=STYLE["text"],
                fontweight="bold" if "PK" in key else "normal")
        if key:
            ax.text(x + WIDTH - 0.12, cy, key, va="center", ha="right", fontsize=7,
                    color=STYLE["pk"] if "PK" in key else STYLE["line"])
    return (x, y - height, WIDTH, height)


def draw(path=None, style=None, title=True):
    """Render the diagram and return the PNG path (style overrides colours)."""
    STYLE.update(style or {})
    path = path or config.FIGURES_DIR / "00_er_diagram.png"
    fig, ax = plt.subplots(figsize=(15.5, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    boxes = {name: _box(ax, name, x, y, cols) for name, (x, y, cols) in TABLES.items()}
    for child, parent, label in RELATIONS:
        cx, cy, cw, ch = boxes[child]
        px, py, pw, ph = boxes[parent]
        start = (cx + cw / 2, cy + ch / 2)
        end = (px + pw / 2, py + ph / 2)
        # Connect the nearest vertical edges.
        if px + pw < cx:
            start, end = (cx, start[1]), (px + pw, end[1])
        elif cx + cw < px:
            start, end = (cx + cw, start[1]), (px, end[1])
        else:
            start = (start[0], cy if py < cy else cy + ch)
            end = (end[0], py + ph if py < cy else py)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": STYLE["line"], "lw": 1.1,
                                "connectionstyle": "arc3,rad=0.05"})
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, fontsize=7,
                color=STYLE["line"], ha="center")
    if title:
            ax.set_title("ER diagram of covtype.db (arrows point from foreign key to referenced table)",
                     fontweight="bold")
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    return path


MERMAID = """erDiagram
    COVER_TYPE ||--o{ CELL : "cover_type_id"
    WILDERNESS_AREA ||--o{ CELL : "wilderness_id"
    SOIL_TYPE ||--o{ CELL : "soil_id"
    CLIMATIC_ZONE ||--o{ SOIL_TYPE : "climatic_zone_id"
    GEOLOGIC_ZONE ||--o{ SOIL_TYPE : "geologic_zone_id"
    CELL ||--|| SPLIT_ASSIGNMENT : "cell_id"
    CELL ||--o{ PREDICTION : "cell_id"
    MODEL ||--o{ PREDICTION : "model_id"
    MODEL ||--o{ EXPERIMENT : "model_id"
    COVER_TYPE ||--o{ PREDICTION : "predicted_cover_type_id"
    CELL {
        int cell_id PK
        int elevation
        int aspect
        int slope
        int horizontal_distance_to_hydrology
        int vertical_distance_to_hydrology
        int horizontal_distance_to_roadways
        int hillshade_9am
        int hillshade_noon
        int hillshade_3pm
        int horizontal_distance_to_fire_points
        int wilderness_id FK
        int soil_id FK
        int cover_type_id FK
    }
    SOIL_TYPE {
        int soil_id PK
        int elu_code
        text description
        int climatic_zone_id FK
        int geologic_zone_id FK
    }
    COVER_TYPE { int cover_type_id PK
        text name }
    WILDERNESS_AREA { int wilderness_id PK
        text name }
    CLIMATIC_ZONE { int climatic_zone_id PK
        text name }
    GEOLOGIC_ZONE { int geologic_zone_id PK
        text name }
    SPLIT_ASSIGNMENT { int cell_id PK
        text split }
    MODEL { int model_id PK
        text name
        text family
        text params_json }
    EXPERIMENT { int experiment_id PK
        int model_id FK
        real accuracy
        real f1_macro
        real balanced_accuracy
        real train_seconds
        real predict_seconds
        real model_size_mb
        real peak_memory_mb
        text created_at }
    PREDICTION { int cell_id PK
        int model_id PK
        int predicted_cover_type_id FK }
"""


if __name__ == "__main__":
    print(draw())
