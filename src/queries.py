"""Analytical SQL queries over the normalised database.

Each entry: title, what it shows, which SQL features it uses, the SQL.
Queries Q1-Q9 describe the data, Q10-Q12 analyse model errors and need
the `prediction` table filled by the training step.
"""

import time

import pandas as pd

from src import config, database

QUERIES = {
    "Q01_class_distribution": {
        "title": "Class distribution with share of total",
        "features": "JOIN, GROUP BY, window SUM() OVER ()",
        "sql": """
            SELECT ct.cover_type_id, ct.name AS cover_type,
                   COUNT(*) AS n_cells,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
            FROM cell c
            JOIN cover_type ct ON ct.cover_type_id = c.cover_type_id
            GROUP BY ct.cover_type_id, ct.name
            ORDER BY n_cells DESC;
        """,
    },
    "Q02_terrain_by_class_and_area": {
        "title": "Average terrain per cover type and wilderness area",
        "features": "2 JOINs, GROUP BY, HAVING",
        "sql": """
            SELECT w.name AS wilderness, ct.name AS cover_type,
                   COUNT(*) AS n_cells,
                   ROUND(AVG(c.elevation), 0) AS avg_elevation,
                   ROUND(AVG(c.slope), 1) AS avg_slope,
                   ROUND(AVG(c.horizontal_distance_to_roadways), 0) AS avg_dist_road
            FROM cell c
            JOIN wilderness_area w ON w.wilderness_id = c.wilderness_id
            JOIN cover_type ct ON ct.cover_type_id = c.cover_type_id
            GROUP BY w.name, ct.name
            HAVING COUNT(*) >= 100
            ORDER BY w.name, avg_elevation DESC;
        """,
    },
    "Q03_dominant_class_per_soil": {
        "title": "Dominant cover type of every frequent soil type",
        "features": "CTE, JOIN, GROUP BY/HAVING, ROW_NUMBER() OVER (PARTITION BY)",
        "sql": """
            WITH soil_class AS (
                SELECT c.soil_id, c.cover_type_id, COUNT(*) AS n
                FROM cell c
                GROUP BY c.soil_id, c.cover_type_id
            ),
            ranked AS (
                SELECT soil_id, cover_type_id, n,
                       SUM(n) OVER (PARTITION BY soil_id) AS soil_total,
                       ROW_NUMBER() OVER (PARTITION BY soil_id ORDER BY n DESC) AS rn
                FROM soil_class
            )
            SELECT r.soil_id, s.elu_code, ct.name AS dominant_cover_type,
                   r.soil_total,
                   ROUND(100.0 * r.n / r.soil_total, 1) AS purity_pct
            FROM ranked r
            JOIN soil_type s ON s.soil_id = r.soil_id
            JOIN cover_type ct ON ct.cover_type_id = r.cover_type_id
            WHERE r.rn = 1 AND r.soil_total >= 1000
            ORDER BY purity_pct DESC;
        """,
    },
    "Q04_elevation_bands": {
        "title": "Cover-type mix by 200 m elevation band",
        "features": "CTE, CASE, GROUP BY, window share within band",
        "sql": """
            WITH banded AS (
                SELECT (elevation / 200) * 200 AS band_start, cover_type_id
                FROM cell
            )
            SELECT b.band_start, b.band_start + 199 AS band_end, ct.name AS cover_type,
                   COUNT(*) AS n_cells,
                   ROUND(100.0 * COUNT(*) /
                         SUM(COUNT(*)) OVER (PARTITION BY b.band_start), 1) AS share_in_band_pct
            FROM banded b
            JOIN cover_type ct ON ct.cover_type_id = b.cover_type_id
            GROUP BY b.band_start, ct.name
            ORDER BY b.band_start, n_cells DESC;
        """,
    },
    "Q05_climatic_zone_vs_class": {
        "title": "Cover types per climatic zone of the soil",
        "features": "3 JOINs through soil_type, GROUP BY",
        "sql": """
            SELECT cz.climatic_zone_id, cz.name AS climatic_zone, ct.name AS cover_type,
                   COUNT(*) AS n_cells
            FROM cell c
            JOIN soil_type s ON s.soil_id = c.soil_id
            JOIN climatic_zone cz ON cz.climatic_zone_id = s.climatic_zone_id
            JOIN cover_type ct ON ct.cover_type_id = c.cover_type_id
            GROUP BY cz.climatic_zone_id, cz.name, ct.name
            ORDER BY cz.climatic_zone_id, n_cells DESC;
        """,
    },
    "Q06_classes_above_overall_mean": {
        "title": "Classes whose mean elevation is above the overall mean, and spread",
        "features": "Subquery in FROM, scalar subquery in WHERE, JOIN, GROUP BY",
        "sql": """
            SELECT ct.name AS cover_type, a.n_cells,
                   ROUND(a.avg_elevation, 0) AS avg_elevation,
                   ROUND(a.avg_elevation - (SELECT AVG(elevation) FROM cell), 0)
                       AS diff_vs_overall_mean,
                   a.min_elevation, a.max_elevation
            FROM (
                SELECT cover_type_id, COUNT(*) AS n_cells, AVG(elevation) AS avg_elevation,
                       MIN(elevation) AS min_elevation, MAX(elevation) AS max_elevation
                FROM cell
                GROUP BY cover_type_id
            ) a
            JOIN cover_type ct ON ct.cover_type_id = a.cover_type_id
            WHERE a.avg_elevation > (SELECT AVG(elevation) FROM cell)
            ORDER BY a.avg_elevation DESC;
        """,
    },
    "Q07_rank_areas_by_road_distance": {
        "title": "Rank of wilderness areas by distance to roads within each class",
        "features": "CTE, JOIN, RANK() OVER (PARTITION BY ... ORDER BY)",
        "sql": """
            WITH stats AS (
                SELECT cover_type_id, wilderness_id,
                       AVG(horizontal_distance_to_roadways) AS avg_road,
                       COUNT(*) AS n
                FROM cell
                GROUP BY cover_type_id, wilderness_id
                HAVING COUNT(*) >= 50
            )
            SELECT ct.name AS cover_type, w.name AS wilderness,
                   ROUND(s.avg_road, 0) AS avg_dist_road, s.n,
                   RANK() OVER (PARTITION BY s.cover_type_id ORDER BY s.avg_road DESC)
                       AS rank_most_remote
            FROM stats s
            JOIN cover_type ct ON ct.cover_type_id = s.cover_type_id
            JOIN wilderness_area w ON w.wilderness_id = s.wilderness_id
            ORDER BY ct.name, rank_most_remote;
        """,
    },
    "Q08_elevation_quartiles_per_area": {
        "title": "Class mix in elevation quartiles inside each wilderness area",
        "features": "Subquery in FROM, NTILE() OVER (PARTITION BY), GROUP BY",
        "sql": """
            SELECT w.name AS wilderness, q.quartile,
                   MIN(q.elevation) AS min_elev, MAX(q.elevation) AS max_elev,
                   ROUND(100.0 * SUM(q.cover_type_id = 1) / COUNT(*), 1) AS spruce_fir_pct,
                   ROUND(100.0 * SUM(q.cover_type_id = 2) / COUNT(*), 1) AS lodgepole_pct,
                   ROUND(100.0 * SUM(q.cover_type_id = 7) / COUNT(*), 1) AS krummholz_pct
            FROM (
                SELECT wilderness_id, elevation, cover_type_id,
                       NTILE(4) OVER (PARTITION BY wilderness_id ORDER BY elevation) AS quartile
                FROM cell
            ) q
            JOIN wilderness_area w ON w.wilderness_id = q.wilderness_id
            GROUP BY w.name, q.quartile
            ORDER BY w.name, q.quartile;
        """,
    },
    "Q09_rare_soils": {
        "title": "Rare soil types (< 1000 cells) and the classes they host",
        "features": "JOIN, GROUP BY, HAVING, GROUP_CONCAT",
        "sql": """
            SELECT s.soil_id, s.elu_code, gz.name AS geologic_zone,
                   COUNT(*) AS n_cells,
                   COUNT(DISTINCT c.cover_type_id) AS n_classes,
                   GROUP_CONCAT(DISTINCT c.cover_type_id) AS class_ids
            FROM cell c
            JOIN soil_type s ON s.soil_id = c.soil_id
            JOIN geologic_zone gz ON gz.geologic_zone_id = s.geologic_zone_id
            GROUP BY s.soil_id
            HAVING COUNT(*) < 1000
            ORDER BY n_cells;
        """,
    },
    "Q10_recall_per_class": {
        "title": "Per-class recall of every model on the test split",
        "features": "4 JOINs, CTE, GROUP BY, conditional aggregation",
        "requires_predictions": True,
        "sql": """
            WITH scored AS (
                SELECT m.name AS model, c.cover_type_id,
                       p.predicted_cover_type_id = c.cover_type_id AS correct
                FROM prediction p
                JOIN cell c ON c.cell_id = p.cell_id
                JOIN model m ON m.model_id = p.model_id
                JOIN split_assignment sa ON sa.cell_id = p.cell_id AND sa.split = 'test'
            )
            SELECT sc.model, ct.name AS cover_type, COUNT(*) AS n_test,
                   ROUND(AVG(sc.correct), 4) AS recall
            FROM scored sc
            JOIN cover_type ct ON ct.cover_type_id = sc.cover_type_id
            GROUP BY sc.model, ct.name
            ORDER BY sc.model, recall;
        """,
    },
    "Q11_top_confusions": {
        "title": "Top-2 confusions per true class for the best model",
        "features": "CTE, JOINs, ROW_NUMBER() OVER (PARTITION BY), subquery",
        "requires_predictions": True,
        "sql": """
            WITH best AS (
                SELECT e.model_id FROM experiment e
                ORDER BY e.f1_macro DESC LIMIT 1
            ),
            errors AS (
                SELECT c.cover_type_id AS true_id, p.predicted_cover_type_id AS pred_id,
                       COUNT(*) AS n
                FROM prediction p
                JOIN cell c ON c.cell_id = p.cell_id
                WHERE p.model_id = (SELECT model_id FROM best)
                  AND p.predicted_cover_type_id <> c.cover_type_id
                GROUP BY true_id, pred_id
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY true_id ORDER BY n DESC) AS rn,
                       SUM(n) OVER (PARTITION BY true_id) AS errors_of_class
                FROM errors
            )
            SELECT t.name AS true_class, pr.name AS predicted_as, r.n,
                   r.errors_of_class
            FROM ranked r
            JOIN cover_type t ON t.cover_type_id = r.true_id
            JOIN cover_type pr ON pr.cover_type_id = r.pred_id
            WHERE r.rn <= 2
            ORDER BY r.errors_of_class DESC, r.n DESC;
        """,
    },
    "Q12_error_rate_by_area_and_band": {
        "title": "Error rate of the best model by wilderness area and elevation band",
        "features": "CTE, JOINs, GROUP BY/HAVING, LAG() OVER",
        "requires_predictions": True,
        "sql": """
            WITH best AS (
                SELECT model_id FROM experiment ORDER BY f1_macro DESC LIMIT 1
            ),
            agg AS (
                SELECT w.name AS wilderness, (c.elevation / 250) * 250 AS band_start,
                       COUNT(*) AS n_test,
                       ROUND(100.0 * AVG(p.predicted_cover_type_id <> c.cover_type_id), 2)
                           AS error_pct
                FROM prediction p
                JOIN cell c ON c.cell_id = p.cell_id
                JOIN wilderness_area w ON w.wilderness_id = c.wilderness_id
                WHERE p.model_id = (SELECT model_id FROM best)
                GROUP BY w.name, band_start
                HAVING COUNT(*) >= 200
            )
            SELECT *, ROUND(error_pct - LAG(error_pct) OVER
                            (PARTITION BY wilderness ORDER BY band_start), 2)
                      AS change_vs_lower_band
            FROM agg
            ORDER BY wilderness, band_start;
        """,
    },
}


def list_queries():
    """Return (key, title, features) for the menu."""
    return [(key, q["title"], q["features"]) for key, q in QUERIES.items()]


def run_query(key, save=True):
    """Run one query by key, return (DataFrame, seconds)."""
    if key not in QUERIES:
        raise KeyError(f"Unknown query {key}. Available: {', '.join(QUERIES)}")
    start = time.perf_counter()
    frame = database.read_sql(QUERIES[key]["sql"])
    seconds = time.perf_counter() - start
    if save:
        config.SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(config.SQL_RESULTS_DIR / f"{key}.csv", index=False)
    return frame, seconds


def run_all(include_predictions=True):
    """Run every query and return a summary table (rows, seconds)."""
    summary = []
    for key, query in QUERIES.items():
        if query.get("requires_predictions") and not include_predictions:
            continue
        frame, seconds = run_query(key)
        summary.append(
            {"query": key, "title": query["title"], "features": query["features"],
             "rows": len(frame), "seconds": round(seconds, 3)}
        )
    return pd.DataFrame(summary)


def pandas_vs_sql():
    """Same aggregation (Q02 without HAVING) in pandas and in SQLite."""
    from src import data_loader

    start = time.perf_counter()
    df = data_loader.load_parquet()
    pandas_result = (
        df.groupby([config.WILDERNESS, config.TARGET])
        .agg(n=("Elevation", "size"), avg_elevation=("Elevation", "mean"))
        .reset_index()
    )
    pandas_seconds = time.perf_counter() - start

    start = time.perf_counter()
    sql_result = database.read_sql(
        """SELECT wilderness_id, cover_type_id, COUNT(*) AS n, AVG(elevation) AS avg_elevation
           FROM cell GROUP BY wilderness_id, cover_type_id"""
    )
    sql_seconds = time.perf_counter() - start

    same = bool(
        (pandas_result["n"].to_numpy() == sql_result["n"].to_numpy()).all()
        and abs(pandas_result["avg_elevation"].to_numpy()
                - sql_result["avg_elevation"].to_numpy()).max() < 1e-6
    )
    return {
        "pandas_seconds_incl_parquet_read": round(pandas_seconds, 3),
        "sqlite_seconds": round(sql_seconds, 3),
        "results_identical": same,
        "groups": len(sql_result),
    }
