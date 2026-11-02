"""Model zoo, hyper-parameter search and resource measurements.

Compared approaches
-------------------
baseline       DummyClassifier (always the majority class)
logreg         multinomial Logistic Regression (linear model)
random_forest  Random Forest (bagging of deep trees)
hist_gb        scikit-learn HistGradientBoosting (gradient boosting)
lightgbm       LightGBM (gradient boosting, leaf-wise trees)
"""

import io
import json
import threading
import time
import warnings

import joblib
import psutil
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import THREADS, config, features

N_JOBS = THREADS


def model_specs():
    """Estimator, preprocessing kind and search space for every model.

    Search spaces are small on purpose: they are explored on a stratified
    subsample with 3-fold CV, which keeps tuning under ~15 minutes on a
    laptop CPU.
    """
    seed = config.RANDOM_STATE
    return {
        "baseline_dummy": {
            "family": "baseline",
            "kind": "tree",
            "estimator": DummyClassifier(strategy="most_frequent"),
            "space": None,
        },
        "logreg": {
            "family": "linear",
            "kind": "linear",
            "estimator": LogisticRegression(max_iter=500, random_state=seed),
            "space": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 8,
        },
        "random_forest": {
            "family": "bagging",
            "kind": "tree",
            "estimator": RandomForestClassifier(random_state=seed, n_jobs=N_JOBS),
            "space": {
                "model__n_estimators": [100, 200],
                "model__max_features": ["sqrt", 0.5],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced_subsample"],
            },
            "n_iter": 8,
        },
        "hist_gb": {
            "family": "boosting",
            "kind": "tree",
            "estimator": HistGradientBoostingClassifier(
                random_state=seed, early_stopping=False,
                # Wilderness and Soil are the last two columns after preprocessing.
                categorical_features=[len(features.NUMERIC_MODEL_COLS),
                                      len(features.NUMERIC_MODEL_COLS) + 1],
            ),
            "space": {
                "model__learning_rate": [0.1, 0.2],
                "model__max_iter": [300, 500],
                "model__max_leaf_nodes": [63, 127, 255],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 5,
        },
        "lightgbm": {
            "family": "boosting",
            "kind": "tree",
            "estimator": LGBMClassifier(random_state=seed, n_jobs=N_JOBS, verbose=-1),
            "space": {
                "model__n_estimators": [500, 1000],
                # lr 0.2 with tiny leaves diverged on rare classes in a pilot run.
                "model__learning_rate": [0.05, 0.1],
                "model__num_leaves": [63, 127, 255],
                "model__min_child_samples": [10, 30],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 6,
        },
    }


def build_pipeline(spec):
    """Preprocessing + estimator in one Pipeline (fit on train only)."""
    pre = features.linear_preprocessor() if spec["kind"] == "linear" else features.tree_preprocessor()
    return Pipeline([("prep", pre), ("model", spec["estimator"])])


class PeakMemory:
    """Context manager: peak extra RSS (MB) while the block runs.

    tracemalloc does not see memory allocated by C++ (LightGBM), so the
    process resident set size is sampled in a background thread instead.
    """

    def __init__(self, interval=0.05):
        self.interval = interval
        self.process = psutil.Process()
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def _watch(self):
        while not self._stop.is_set():
            current = self.process.memory_info().rss / 1024**2
            self.peak_mb = max(self.peak_mb, current - self.baseline)
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self.process.memory_info().rss / 1024**2
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()
        return False


def model_size_mb(model):
    """Size of the pickled model in memory (what joblib would write)."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getbuffer().nbytes / 1024**2


def tune(name, X_sub, y_sub, verbose=True):
    """RandomizedSearchCV with stratified K-fold on the subsample.

    Returns best params (dict, keys without the 'model__' prefix) and a
    small table of all tried candidates.
    """
    spec = model_specs()[name]
    if spec["space"] is None:
        return {}, []
    search = RandomizedSearchCV(
        build_pipeline(spec),
        spec["space"],
        n_iter=spec["n_iter"],
        scoring="f1_macro",
        cv=StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE,
        # Models that are already parallel run their CV folds one by one.
        n_jobs=1 if name in ("random_forest", "lightgbm", "hist_gb") else N_JOBS,
        refit=False,
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings of weak C values
        search.fit(X_sub, y_sub)
    seconds = time.perf_counter() - start
    results = search.cv_results_
    candidates = []
    for i in range(len(results["params"])):
        candidates.append(
            {
                "params": {k.replace("model__", ""): v for k, v in results["params"][i].items()},
                "cv_f1_macro_mean": round(float(results["mean_test_score"][i]), 4),
                "cv_f1_macro_std": round(float(results["std_test_score"][i]), 4),
                "fit_seconds_mean": round(float(results["mean_fit_time"][i]), 2),
            }
        )
    candidates.sort(key=lambda c: c["cv_f1_macro_mean"], reverse=True)
    best = candidates[0]["params"]
    if verbose:
        print(f"  {name}: best CV F1-macro {candidates[0]['cv_f1_macro_mean']:.4f} "
              f"in {seconds:.0f}s with {best}")
    return best, candidates


def fit_final(name, params, X_train, y_train):
    """Fit the pipeline with chosen params on the full training split."""
    spec = model_specs()[name]
    pipeline = build_pipeline(spec)
    if params:
        pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})
    with PeakMemory() as memory, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
    return pipeline, {"train_seconds": round(train_seconds, 2),
                      "peak_memory_mb": round(memory.peak_mb, 1)}


def save_model(pipeline, meta, path=None):
    """Persist the fitted pipeline with joblib plus a JSON description."""
    path = path or config.BEST_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    config.BEST_MODEL_META.write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_model(path=None):
    """Load the saved pipeline; clear message if training was not run."""
    path = path or config.BEST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the models first (menu step 5).")
    return joblib.load(path)
