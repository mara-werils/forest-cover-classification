"""High-level steps that glue the modules together.

The console menu (main.py) calls these functions one by one;
`python main.py --all` runs them in order for full reproduction.
"""

import json
import time
from contextlib import closing

import pandas as pd

from src import (
    audit,
    config,
    data_loader,
    database,
    eda,
    evaluate,
    features,
    models,
    queries,
    robustness,
)

MODEL_ORDER = ["baseline_dummy", "logreg", "random_forest", "hist_gb", "lightgbm"]


def step_prepare(force=False):
    """Download, chunked typed read, memory report, Parquet, CSV-vs-Parquet."""
    config.ensure_dirs()
    df, memory = data_loader.prepare_data(force=force)
    bench = data_loader.benchmark_formats()
    print(f"Rows: {len(df):,}. Memory: {memory['default_read']['memory_mb']} MB -> "
          f"{memory['chunked_typed_read']['memory_mb']} MB "
          f"(-{memory['memory_reduction_pct']}%)")
    print(bench.to_string(index=False))
    return df


def step_audit(df=None):
    df = df if df is not None else data_loader.load_parquet()
    report = audit.run_audit(df)
    audit.print_audit(report)
    return report


def step_eda(df=None):
    df = df if df is not None else data_loader.load_parquet()
    report = json.loads(config.AUDIT_JSON.read_text()) if config.AUDIT_JSON.exists() else None
    outputs = eda.run_eda(df, report)
    print(f"Saved {len(outputs) - 1} figures to {config.FIGURES_DIR}")
    print("Strongest correlations:", outputs["top_correlations"])
    return outputs


def step_database(df=None):
    df = df if df is not None else data_loader.load_parquet()
    counts = database.build_database(df)
    print("Rows per table:", counts)
    summary = queries.run_all(include_predictions=False)
    print(summary[["query", "rows", "seconds"]].to_string(index=False))
    comparison = queries.pandas_vs_sql()
    print("pandas vs SQL:", comparison)
    (config.REPORTS_DIR / "pandas_vs_sql.json").write_text(json.dumps(comparison, indent=2))
    return counts


def _record(connection, name, family, params, metrics, test_index, predictions):
    """Store one experiment in SQLite (model, metrics, predictions)."""
    model_id = database.register_model(connection, name, family, params)
    database.save_experiment(connection, model_id, metrics)
    database.save_predictions(connection, model_id, test_index, predictions)


def step_train(df=None, model_names=None, tuning_sample=config.TUNING_SAMPLE):
    """Tune on a subsample with CV, refit on the full train split, evaluate.

    Returns the results DataFrame (one row per model).
    """
    df = df if df is not None else data_loader.load_parquet()
    model_names = model_names or MODEL_ORDER
    X_train, X_test, y_train, y_test = features.split_data(df)
    print(f"Train: {len(X_train):,} rows, test: {len(X_test):,} rows (stratified 80/20)")

    with closing(database.connect()) as connection:
        database.save_split(connection, X_train.index, X_test.index)

    X_sub, y_sub = features.stratified_subsample(X_train, y_train, tuning_sample)
    specs = models.model_specs()
    tuning_log = {}
    rows = []
    fitted = {}
    for name in model_names:
        print(f"\n[{name}] tuning on {len(X_sub):,} rows, {config.CV_FOLDS}-fold CV ...")
        start = time.perf_counter()
        best_params, candidates = models.tune(name, X_sub, y_sub)
        tuning_seconds = time.perf_counter() - start
        tuning_log[name] = {"best_params": best_params, "seconds": round(tuning_seconds, 1),
                            "candidates": candidates}

        print(f"[{name}] fitting on full train split ...")
        pipeline, cost = models.fit_final(name, best_params, X_train, y_train)
        metrics, predictions = evaluate.score(pipeline, X_test, y_test)
        metrics.update(cost)
        metrics["model_size_mb"] = round(models.model_size_mb(pipeline), 2)
        metrics["tuning_seconds"] = round(tuning_seconds, 1)
        row = {"model": name, "family": specs[name]["family"], **metrics,
               "params": json.dumps(best_params, default=str)}
        rows.append(row)
        fitted[name] = (pipeline, predictions, best_params)
        print(f"[{name}] F1-macro={metrics['f1_macro']:.4f} acc={metrics['accuracy']:.4f} "
              f"train={metrics['train_seconds']}s size={metrics['model_size_mb']} MB")
        with closing(database.connect()) as connection:
            _record(connection, name, specs[name]["family"], best_params, metrics,
                    X_test.index, predictions)

    # Imbalance experiment: the best LightGBM with the opposite class weighting.
    if "lightgbm" in fitted:
        params = dict(fitted["lightgbm"][2])
        params["class_weight"] = None if params.get("class_weight") == "balanced" else "balanced"
        variant = "lightgbm_balanced" if params["class_weight"] else "lightgbm_unweighted"
        print(f"\n[{variant}] imbalance check: same params, class_weight={params['class_weight']}")
        pipeline, cost = models.fit_final("lightgbm", params, X_train, y_train)
        metrics, predictions = evaluate.score(pipeline, X_test, y_test)
        metrics.update(cost)
        metrics["model_size_mb"] = round(models.model_size_mb(pipeline), 2)
        metrics["tuning_seconds"] = 0.0
        rows.append({"model": variant, "family": "boosting", **metrics,
                     "params": json.dumps(params, default=str)})
        fitted[variant] = (pipeline, predictions, params)
        with closing(database.connect()) as connection:
            _record(connection, variant, "boosting", params, metrics, X_test.index, predictions)

    results = pd.DataFrame(rows)
    # Re-running a subset of models keeps the rows of the other models.
    if set(model_names) != set(MODEL_ORDER) and config.RESULTS_PARQUET.exists():
        previous = pd.read_parquet(config.RESULTS_PARQUET)
        previous = previous[~previous["model"].isin(results["model"])]
        results = pd.concat([previous, results], ignore_index=True)
        order = MODEL_ORDER + ["lightgbm_balanced", "lightgbm_unweighted"]
        results = results.sort_values("model", key=lambda s: s.map(order.index)).reset_index(
            drop=True)
        if config.TUNING_JSON.exists():
            tuning_log = {**json.loads(config.TUNING_JSON.read_text()), **tuning_log}
    results.to_parquet(config.RESULTS_PARQUET, index=False)
    results.to_csv(config.RESULTS_CSV, index=False)
    config.TUNING_JSON.write_text(json.dumps(tuning_log, indent=2, default=str))

    best_name = results.loc[results["f1_macro"].idxmax(), "model"]
    evaluate.plot_model_comparison(results)
    evaluate.plot_per_class_f1(results)
    if best_name not in fitted:
        print(f"\nBest model is still {best_name} (saved earlier); comparison plots updated.")
        return results
    best_pipeline, best_predictions, best_params = fitted[best_name]
    models.save_model(best_pipeline, {
        "model": best_name,
        "params": best_params,
        "metrics": results.loc[results["model"] == best_name].iloc[0].drop("params").to_dict(),
        "input_columns": features.INPUT_COLS,
        "trained_on_rows": len(X_train),
    })

    evaluate.plot_confusion(y_test, best_predictions,
                            f"Confusion matrix: {best_name} (test, row %)",
                            "10_confusion_best.png")
    if "logreg" in fitted:
        evaluate.plot_confusion(y_test, fitted["logreg"][1],
                                "Confusion matrix: logistic regression (test, row %)",
                                "10b_confusion_logreg.png")
    evaluate.report_table(y_test, best_predictions).to_csv(
        config.REPORTS_DIR / "classification_report_best.csv")

    print("\n" + results[["model", "accuracy", "f1_macro", "balanced_accuracy",
                          "train_seconds", "model_size_mb", "peak_memory_mb"]].to_string(index=False))
    print(f"\nBest model by F1-macro: {best_name} -> saved to {config.BEST_MODEL_PATH}")
    return results


def step_interpret(df=None, learning_curve=True):
    """Permutation importance, error analysis and learning curve."""
    df = df if df is not None else data_loader.load_parquet()
    X_train, X_test, y_train, y_test = features.split_data(df)
    model = models.load_model()
    meta = json.loads(config.BEST_MODEL_META.read_text())

    importance = evaluate.permutation_importance_table(model, X_test, y_test)
    importance.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)
    evaluate.plot_importance(importance, meta["model"])
    print(importance.to_string(index=False))

    predictions = model.predict(X_test)
    analysis = evaluate.error_analysis(model, X_test, y_test, predictions)
    evaluate.plot_error_profile(analysis)
    for key in ("by_elevation_band", "by_wilderness", "confidence", "confident_mistakes"):
        analysis[key].to_csv(config.REPORTS_DIR / f"errors_{key}.csv")
    (config.REPORTS_DIR / "errors_summary.json").write_text(
        json.dumps(analysis["summary"], indent=2))
    print("Error summary:", analysis["summary"])
    print(analysis["by_wilderness"].to_string())

    if learning_curve:
        curve = run_learning_curve(X_train, y_train, X_test, y_test)
        curve.to_csv(config.REPORTS_DIR / "learning_curve.csv", index=False)
        evaluate.plot_learning_curve(curve)
        print(curve.to_string(index=False))
    return importance, analysis


def run_learning_curve(X_train, y_train, X_test, y_test,
                       sizes=(5_000, 20_000, 80_000, 200_000)):
    """F1-macro of tuned models trained on growing stratified subsets.

    The full-size point is taken from the main experiment table.
    """
    tuning = json.loads(config.TUNING_JSON.read_text())
    results = pd.read_parquet(config.RESULTS_PARQUET)
    rows = []
    for name in ("logreg", "random_forest", "lightgbm"):
        params = tuning[name]["best_params"]
        for size in sizes:
            X_part, y_part = features.stratified_subsample(X_train, y_train, size)
            pipeline, cost = models.fit_final(name, params, X_part, y_part)
            metrics, _ = evaluate.score(pipeline, X_test, y_test)
            rows.append({"model": name, "train_rows": size, "f1_macro": metrics["f1_macro"],
                         "train_seconds": cost["train_seconds"]})
            print(f"  learning curve {name} @ {size:,}: F1-macro {metrics['f1_macro']:.4f}")
        full = results.loc[results["model"] == name].iloc[0]
        rows.append({"model": name, "train_rows": len(X_train), "f1_macro": full["f1_macro"],
                     "train_seconds": full["train_seconds"]})
    return pd.DataFrame(rows)


def step_sql_report():
    """Run all 12 queries (including model-error queries) and save CSVs."""
    summary = queries.run_all(include_predictions=True)
    summary.to_csv(config.SQL_RESULTS_DIR / "_summary.csv", index=False)
    print(summary[["query", "features", "rows", "seconds"]].to_string(index=False))
    return summary


def run_all():
    """Full reproduction: data -> audit -> EDA -> DB -> models -> analysis."""
    start = time.perf_counter()
    df = step_prepare()
    step_audit(df)
    step_eda(df)
    step_database(df)
    step_train(df)
    step_interpret(df)
    step_sql_report()
    robustness.run_robustness(df)
    print(f"\nPipeline finished in {(time.perf_counter() - start) / 60:.1f} min")
