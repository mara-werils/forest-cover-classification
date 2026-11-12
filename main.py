"""Forest Cover Type Classification: interactive console menu.

Usage
-----
python main.py          # interactive menu
python main.py --all    # run the whole pipeline without questions
"""

import sys

from src import ablation, config, pipeline, predict, queries, robustness, significance

MENU = {
    "1": "Prepare data (download, chunked load, dtype optimisation, Parquet)",
    "2": "Data quality audit",
    "3": "Build SQLite database (normalised schema)",
    "4": "Exploratory analysis: save all figures",
    "5": "Train and compare models (baseline, LR, RF, HistGB, LightGBM)",
    "6": "Interpret best model (importance, error analysis, learning curve)",
    "7": "Run an SQL query",
    "8": "Predict cover type for your own values",
    "9": "Show experiment results",
    "10": "Robustness: ROC/PR-AUC + unseen-area validation (~25 min)",
    "11": "Statistics: feature relevance + shift between wilderness areas",
    "12": "Feature ablation: do the engineered features help? (~25 min)",
    "13": "Significance: bootstrap CIs + McNemar tests between models (~15 min)",
    "0": "Exit",
}


def print_menu():
    print("\n" + "=" * 64)
    print(" FOREST COVER TYPE CLASSIFICATION ".center(64, "="))
    print("=" * 64)
    for key, label in MENU.items():
        print(f" {key} - {label}")


def ask_int(prompt, default, low, high):
    """Ask for an integer until the user enters a valid one."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if low <= value <= high:
            return value
        print(f"  Value must be between {low} and {high}.")


def sql_menu():
    """List the queries and run the selected one."""
    items = queries.list_queries()
    for number, (_key, title, used) in enumerate(items, start=1):
        print(f" {number:>2}. {title}  [{used}]")
    choice = ask_int("Query number", 1, 1, len(items))
    key = items[choice - 1][0]
    frame, seconds = queries.run_query(key)
    print(frame.head(30).to_string(index=False))
    print(f"({len(frame)} rows, {seconds:.3f} s, saved to {config.SQL_RESULTS_DIR / key}.csv)")


def prediction_menu():
    """Ask the user for each feature and print the top-3 cover types."""
    print("Enter values (press Enter to keep the example value).")
    print("Wilderness: " + ", ".join(f"{k}={v}" for k, v in config.WILDERNESS_AREAS.items()))
    record = {}
    for column, (low, high) in predict.INPUT_LIMITS.items():
        record[column] = ask_int(f"  {column}", predict.EXAMPLE_RECORD[column], low, high)
    result = predict.predict_record(record)
    print("\nPrediction:")
    for rank, item in enumerate(result, start=1):
        bar = "#" * int(item["probability"] * 30)
        print(f" {rank}. {item['cover_type']:<18} {item['probability']:>6.1%} {bar}")
    total = predict.append_history(record, result)
    print(f"(saved, {total} predictions in {config.HISTORY_JSON.name})")


def show_results():
    import pandas as pd

    if not config.RESULTS_PARQUET.exists():
        print("No results yet: run step 5 first.")
        return
    results = pd.read_parquet(config.RESULTS_PARQUET)
    columns = ["model", "accuracy", "f1_macro", "balanced_accuracy", "train_seconds",
               "predict_ms_per_1000", "model_size_mb"]
    # ROC-AUC / PR-AUC appear after menu option 10 has been run.
    columns += [c for c in ("roc_auc_ovr_macro", "pr_auc_macro") if c in results.columns]
    print(results[columns].sort_values("f1_macro", ascending=False).to_string(index=False))


ACTIONS = {
    "1": pipeline.step_prepare,
    "2": pipeline.step_audit,
    "3": pipeline.step_database,
    "4": pipeline.step_eda,
    "5": pipeline.step_train,
    "6": pipeline.step_interpret,
    "7": sql_menu,
    "8": prediction_menu,
    "9": show_results,
    "10": robustness.run_robustness,
    "11": pipeline.step_stats,
    "12": ablation.run_ablation,
    "13": significance.run_significance,
}


def main():
    config.ensure_dirs()
    if "--all" in sys.argv:
        pipeline.run_all()
        return
    while True:
        print_menu()
        choice = input("Choose an option: ").strip()
        if choice == "0":
            print("Bye!")
            break
        action = ACTIONS.get(choice)
        if action is None:
            print("Unknown option, try again.")
            continue
        try:
            action()
        except FileNotFoundError as error:
            # Typical case: a step is run before the step that creates its input.
            print(f"[missing file] {error}")
        except ValueError as error:
            print(f"[invalid input] {error}")
        except KeyboardInterrupt:
            print("\nInterrupted, back to menu.")


if __name__ == "__main__":
    main()
