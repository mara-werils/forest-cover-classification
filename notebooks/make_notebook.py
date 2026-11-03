"""Generate Forest_Cover_Defense.ipynb (run from the repository root).

The notebook is a thin, step-by-step front end over the modules in src/:
every cell calls project functions, so the logic stays in one place.
"""

from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/mara-werils/forest-cover-classification"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(f"""
# Forest Cover Type Classification: defense notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mara-werils/forest-cover-classification/blob/main/notebooks/Forest_Cover_Defense.ipynb)

Topic 37, Programming for AI. Team: **Amanbaev Marlen (AAI-2601M)**, **Aktaev Erik (AAI-2601M)**,
**Gajiyev Rashad (AAI-2601M)**.

This notebook walks through the project in the order of the pipeline. It does not contain
its own logic: each cell calls functions from `src/`, the same ones used by the console menu
(`main.py`) and the Streamlit app (`app.py`).

| Step | Module |
|---|---|
| 1. Data: download, chunked typed read, Parquet | `src/data_loader.py` |
| 2. Data quality audit | `src/audit.py` |
| 3. EDA | `src/eda.py` |
| 4. SQLite + SQL | `src/database.py`, `src/queries.py` |
| 5. Models: comparison | `src/features.py`, `src/models.py`, `src/pipeline.py` |
| 6. Evaluation, importance, errors | `src/evaluate.py` |
| 7. Robustness: ROC/PR-AUC, unseen-area validation | `src/robustness.py` |
| 8. Prediction for user input | `src/predict.py` |

Repository: {REPO}
""")

code(f"""
# Setup: in Colab clone the repository, locally just move to the repo root.
import os, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not Path("forest-cover-classification").exists():
        !git clone -q {REPO}.git
    %cd forest-cover-classification
    !pip install -q -r requirements.txt
else:
    root = Path.cwd()
    while not (root / "main.py").exists() and root != root.parent:
        root = root.parent
    os.chdir(root)
sys.path.insert(0, os.getcwd())
print("Working directory:", os.getcwd())
""")

code("""
import json
import pandas as pd
from IPython.display import Image, display
from src import config, data_loader, audit, eda, database, queries, features, models, evaluate, predict

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
config.ensure_dirs()
FIG = config.FIGURES_DIR
""")

md("""
## 1. Data: 581,012 rows, memory-aware loading

The raw file has no header and 55 integer columns. `pd.read_csv` defaults to `int64` for all of
them. We read it in chunks of 100,000 rows with the smallest lossless dtype per column
(`int16` for elevation and distances, `uint8` for flags, angles and hillshade), then collapse
the 4 + 40 one-hot columns into two categorical columns and save Parquet.
""")

code("""
df, memory = data_loader.prepare_data()
print(f"Rows: {len(df):,}, columns: {df.shape[1]}")
print(f"Default read : {memory['default_read']['memory_mb']:>7} MB  ({memory['default_read']['dtypes']})")
print(f"Chunked+typed: {memory['chunked_typed_read']['memory_mb']:>7} MB  -> -{memory['memory_reduction_pct']}%")
pd.DataFrame(memory["chunked_typed_read"]["chunks"])
""")

code("""
# CSV vs Parquet: size on disk, load time, peak RAM (approach comparison, requirement 9)
data_loader.benchmark_formats(repeats=2)
""")

md("## 2. Data quality audit")

code("""
report = audit.run_audit(df)
audit.print_audit(report)
""")

md("""
**Decisions.** No missing values, no duplicates, no rows with zero or several soil/wilderness
flags, no values outside physical ranges. IQR "outliers" (up to 5.4% for vertical distance
to water and fire points) are real terrain, not errors, so they are **kept**. Tree models are
insensitive to them, and the linear model sees them after standard scaling. 1,338 cells have
`Hillshade_3pm = 0` (full shadow), which is plausible on steep north-west slopes, so they are
also kept and flagged in the report.
""")

md("## 3. Exploratory analysis")

code("""
outputs = eda.run_eda(df, report)
print("Strongest correlations:", outputs["top_correlations"])
for name in ["01_class_distribution.png", "05_elevation_by_class.png", "06_wilderness_vs_class.png",
             "04_correlation_heatmap.png", "03_boxplots_by_class.png"]:
    display(Image(filename=str(FIG / name), width=900))
""")

code("""
# Interactive Plotly chart (also saved as reports/figures/09_interactive_scatter.html)
import plotly.express as px
sample = df.sample(8000, random_state=0).assign(Cover=lambda d: d[config.TARGET].map(config.COVER_TYPES))
px.scatter(sample, x="Horizontal_Distance_To_Roadways", y="Elevation", color="Cover", opacity=0.6,
           color_discrete_map={config.COVER_TYPES[k]: v for k, v in eda.CLASS_COLORS.items()},
           template="plotly_white", title="Elevation vs distance to roads")
""")

md("## 4. SQLite: normalised schema and SQL analytics")

code("""
print(database.build_database(df))
display(Image(filename=str(FIG / "00_er_diagram.png"), width=1000))
""")

code("""
summary = queries.run_all(include_predictions=False)
summary
""")

code("""
for key in ["Q01_class_distribution", "Q03_dominant_class_per_soil", "Q07_rank_areas_by_road_distance",
            "Q08_elevation_quartiles_per_area"]:
    frame, seconds = queries.run_query(key)
    print(f"--- {key}: {queries.QUERIES[key]['title']} ({queries.QUERIES[key]['features']})")
    display(frame.head(10))
""")

code("""
# The same aggregation in pandas and in SQLite gives identical numbers
queries.pandas_vs_sql()
""")

md("""
## 5. Models: baseline vs Logistic Regression vs Random Forest vs Gradient Boosting

* Split: stratified 80/20 **before** any fitting; all preprocessing lives inside the
  scikit-learn `Pipeline`, so it is fitted on the training part only.
* Tuning: `RandomizedSearchCV`, 3-fold stratified CV, scoring = F1-macro, on a stratified
  40,000-row subsample of the training split; the best configuration is then refitted on all
  464,809 training rows.
* The full search takes 30–60 min (`python main.py --all`). Its results were saved to
  `reports/experiments.csv` and are loaded here.
""")

code("""
results = pd.read_csv(config.RESULTS_CSV)
cols = ["model", "accuracy", "f1_macro", "balanced_accuracy", "train_seconds",
        "predict_ms_per_1000", "model_size_mb", "peak_memory_mb", "tuning_seconds"]
results[cols].sort_values("f1_macro", ascending=False)
""")

code("""
tuning = json.loads(config.TUNING_JSON.read_text())
pd.DataFrame([{"model": m, "best_params": t["best_params"], "cv_f1_macro": t["candidates"][0]["cv_f1_macro_mean"]
               if t["candidates"] else None, "tuning_s": t["seconds"]} for m, t in tuning.items()])
""")

code("""
for name in ["11_model_comparison.png", "12_per_class_f1.png", "15_learning_curve.png"]:
    if (FIG / name).exists():
        display(Image(filename=str(FIG / name), width=1000))
""")

code("""
# Optional: retrain the best model with its tuned parameters (about 3 min on 4 CPU threads).
# The repository already contains models/best_model.joblib, so this is off by default.
RETRAIN = False
X_train, X_test, y_train, y_test = features.split_data(df)
if RETRAIN or not config.BEST_MODEL_PATH.exists():
    meta = json.loads(config.BEST_MODEL_META.read_text())
    name = "lightgbm" if meta["model"].startswith("lightgbm") else meta["model"]
    model, cost = models.fit_final(name, meta["params"], X_train, y_train)
    models.save_model(model, meta)
    print("Retrained:", cost)
model = models.load_model()
model
""")

md("## 6. Evaluation of the best model on the hold-out test split")

code("""
metrics, predictions = evaluate.score(model, X_test, y_test)
print(json.dumps(metrics, indent=1))
evaluate.report_table(y_test, predictions)
""")

code("""
evaluate.plot_confusion(y_test, predictions, "Confusion matrix: best model (test, row %)", "10_confusion_best.png")
display(Image(filename=str(FIG / "10_confusion_best.png"), width=650))
display(Image(filename=str(FIG / "10b_confusion_logreg.png"), width=650))
""")

code("""
# Permutation importance (precomputed by menu step 6: 20k test rows x 5 repeats)
display(pd.read_csv(config.REPORTS_DIR / "permutation_importance.csv"))
display(Image(filename=str(FIG / "13_permutation_importance.png"), width=750))
""")

code("""
analysis = evaluate.error_analysis(model, X_test, y_test, predictions)
print(analysis["summary"])
display(analysis["by_wilderness"])
display(analysis["confidence"])
display(analysis["confident_mistakes"].head(10))
""")

code("""
# Store predictions in SQLite and analyse the errors with SQL (Q10-Q12)
from contextlib import closing
with closing(database.connect()) as con:
    database.save_split(con, X_train.index, X_test.index)
    model_id = database.register_model(con, "lightgbm", "boosting", {})
    database.save_experiment(con, model_id, {**metrics, "train_seconds": None, "model_size_mb": None})
    database.save_predictions(con, model_id, X_test.index, predictions)
for key in ["Q11_top_confusions", "Q12_error_rate_by_area_and_band"]:
    print("---", queries.QUERIES[key]["title"])
    display(queries.run_query(key)[0].head(12))
""")

md("""
## 7. Robustness: threshold-free metrics and an unseen wilderness area

* **ROC-AUC / PR-AUC** (macro, one-vs-rest) complement F1-macro: they do not depend on the
  0.5 threshold, and PR-AUC is sensitive to the rare classes.
* **Group validation.** The random split puts neighbouring 30 m cells into both train and test.
  The data has no coordinates, so the wilderness area is used as the group: train on three
  areas, test on the fourth (`src/robustness.py`, menu option 10, about 25 min). Results were
  saved by that run and are loaded here.
""")

code("""
from src import robustness
display(pd.read_csv(robustness.PROB_CSV))
group = pd.read_csv(robustness.GROUP_CSV)
display(group.pivot(index="held_out_area", columns="model", values="f1_macro").round(3))
display(Image(filename=str(FIG / "18_group_validation.png"), width=900))
""")

md("## 8. Prediction for user input")

code("""
record = dict(predict.EXAMPLE_RECORD)
print(record)
predict.predict_record(record, model=model)
""")

code("""
# A high, rocky cell in Neota: expected Krummholz / Spruce-Fir
record.update({"Elevation": 3450, "Slope": 18, config.WILDERNESS: 2, config.SOIL: 38,
               "Horizontal_Distance_To_Roadways": 1500})
predict.predict_record(record, model=model)
""")

code("""
# Invalid input is rejected with a clear message (try/except, requirement 6)
try:
    predict.predict_record({**record, "Slope": 140}, model=model)
except ValueError as error:
    print("Rejected:", error)
""")

md("""
## Conclusions

* Gradient boosting (LightGBM) is the best model: F1-macro ≈ 0.956 and accuracy ≈ 0.974 on
  116,203 test cells, against 0.094 for the majority baseline and 0.548 for Logistic Regression.
* Random Forest is almost as accurate (0.949) and trains faster, but its model is 6× larger.
* The classes are not linearly separable: the linear model cannot express "elevation belt ×
  soil × area" interactions, so it loses 40 points of F1-macro, mostly on rare classes.
* Class weighting is a small but consistent gain for the rare classes (balanced accuracy
  +1.2 points) at no cost in accuracy.
* About 69% of the remaining errors are Spruce/Fir ↔ Lodgepole Pine confusions, and most errors
  lie at 2,750–3,250 m, where both species grow on the same terrain.
* Limitation: the random split ignores spatial autocorrelation between neighbouring cells, so
  the score is optimistic for a completely new region.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
out = Path(__file__).resolve().parent / "Forest_Cover_Defense.ipynb"
nbf.write(nb, out)
print(out)
