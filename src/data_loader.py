"""Data acquisition and memory-aware loading.

Steps:
1. download the UCI archive (once) and unpack it to a CSV with a header;
2. read the CSV in chunks with compact dtypes and measure memory
   before/after the optimisation;
3. collapse the 4 + 40 one-hot columns into two categorical columns;
4. store the result as Parquet and compare CSV vs Parquet.
"""

import gzip
import json
import shutil
import time
import tracemalloc
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src import config


def download_dataset(force=False):
    """Download and unpack the Covertype archive. Returns the CSV path."""
    config.ensure_dirs()
    if config.RAW_CSV.exists() and not force:
        return config.RAW_CSV

    if not config.RAW_GZ.exists() or force:
        if not config.RAW_ZIP.exists() or force:
            print(f"Downloading {config.DATA_URL} ...")
            try:
                urllib.request.urlretrieve(config.DATA_URL, config.RAW_ZIP)
            except OSError as error:
                raise RuntimeError(
                    "Download failed. Get the file manually from "
                    f"{config.DATA_SOURCE_PAGE} and put covtype.data.gz "
                    f"into {config.RAW_DIR}"
                ) from error
        with zipfile.ZipFile(config.RAW_ZIP) as archive:
            archive.extract("covtype.data.gz", config.RAW_DIR)

    # The original file has no header; we add one so that the CSV is
    # self-describing and the CSV/Parquet benchmark is fair.
    with gzip.open(config.RAW_GZ, "rt") as src, open(config.RAW_CSV, "w") as dst:
        dst.write(",".join(config.RAW_COLUMNS) + "\n")
        shutil.copyfileobj(src, dst)
    return config.RAW_CSV


def optimized_dtypes():
    """Smallest dtype that holds every column without loss.

    Elevation (1859-3858), distances (< 7200) and the signed vertical
    distance fit into int16; angles, hillshade, flags and the target
    fit into uint8.
    """
    dtypes = {}
    for col in config.NUMERIC_COLS:
        dtypes[col] = "int16"
    for col in ("Aspect", "Slope", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"):
        dtypes[col] = "uint16" if col == "Aspect" else "uint8"
    for col in config.WILDERNESS_COLS + config.SOIL_COLS:
        dtypes[col] = "uint8"
    dtypes[config.TARGET] = "uint8"
    return dtypes


def memory_mb(df):
    """Deep memory usage of a DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / 1024**2


def read_in_chunks(path=None, chunksize=config.CHUNK_SIZE):
    """Read the CSV chunk by chunk with compact dtypes.

    Returns the concatenated frame and a list with per-chunk statistics.
    """
    path = path or config.RAW_CSV
    chunks = []
    stats = []
    reader = pd.read_csv(path, chunksize=chunksize, dtype=optimized_dtypes())
    for number, chunk in enumerate(reader, start=1):
        chunks.append(chunk)
        stats.append(
            {"chunk": number, "rows": len(chunk), "memory_mb": round(memory_mb(chunk), 2)}
        )
    return pd.concat(chunks, ignore_index=True), stats


def collapse_one_hot(df):
    """Add Wilderness (1-4) and Soil (1-40) columns built from one-hot blocks.

    A row with no active flag gets 0, a row with several flags gets the
    first one; the audit counts both cases separately.
    """
    wild = df[config.WILDERNESS_COLS].to_numpy()
    soil = df[config.SOIL_COLS].to_numpy()
    df[config.WILDERNESS] = np.where(wild.any(axis=1), wild.argmax(axis=1) + 1, 0)
    df[config.SOIL] = np.where(soil.any(axis=1), soil.argmax(axis=1) + 1, 0)
    df[config.WILDERNESS] = df[config.WILDERNESS].astype("uint8")
    df[config.SOIL] = df[config.SOIL].astype("uint8")
    return df


def measure_memory(path=None):
    """Compare the default pandas read with the chunked, typed read."""
    path = path or config.RAW_CSV

    start = time.perf_counter()
    naive = pd.read_csv(path)
    naive_time = time.perf_counter() - start
    naive_mb = memory_mb(naive)
    naive_dtypes = naive.dtypes.astype(str).value_counts().to_dict()
    del naive

    start = time.perf_counter()
    optimized, chunk_stats = read_in_chunks(path)
    optimized_time = time.perf_counter() - start
    optimized_mb = memory_mb(optimized)

    report = {
        "rows": int(len(optimized)),
        "columns": int(optimized.shape[1]),
        "default_read": {
            "memory_mb": round(naive_mb, 2),
            "seconds": round(naive_time, 2),
            "dtypes": naive_dtypes,
        },
        "chunked_typed_read": {
            "memory_mb": round(optimized_mb, 2),
            "seconds": round(optimized_time, 2),
            "dtypes": optimized.dtypes.astype(str).value_counts().to_dict(),
            "chunksize": config.CHUNK_SIZE,
            "chunks": chunk_stats,
        },
        "memory_reduction_pct": round(100 * (1 - optimized_mb / naive_mb), 1),
    }
    config.MEMORY_JSON.write_text(json.dumps(report, indent=2))
    return optimized, report


def save_parquet(df, path=None):
    """Write the optimised frame to Parquet (snappy compression)."""
    path = path or config.PARQUET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    return path


def load_parquet(path=None, columns=None):
    """Load the processed data set; raise a clear error if it is missing."""
    path = path or config.PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run step 1 (prepare data) of the menu first."
        )
    return pd.read_parquet(path, columns=columns)


def _timed_read(reader):
    """Run a read function, return (frame, seconds, peak traced MB)."""
    tracemalloc.start()
    start = time.perf_counter()
    frame = reader()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return frame, seconds, peak / 1024**2


def benchmark_formats(repeats=3):
    """CSV (default and typed) vs Parquet: size on disk, load time, peak RAM."""
    readers = {
        "csv_default": lambda: pd.read_csv(config.RAW_CSV),
        "csv_typed": lambda: pd.read_csv(config.RAW_CSV, dtype=optimized_dtypes()),
        "parquet": lambda: pd.read_parquet(config.PARQUET_PATH),
        "parquet_3_columns": lambda: pd.read_parquet(
            config.PARQUET_PATH, columns=["Elevation", config.SOIL, config.TARGET]
        ),
    }
    sizes = {
        "csv_default": config.RAW_CSV.stat().st_size,
        "csv_typed": config.RAW_CSV.stat().st_size,
        "parquet": config.PARQUET_PATH.stat().st_size,
        "parquet_3_columns": config.PARQUET_PATH.stat().st_size,
    }
    results = []
    for name, reader in readers.items():
        times, peaks = [], []
        for _ in range(repeats):
            frame, seconds, peak = _timed_read(reader)
            times.append(seconds)
            peaks.append(peak)
        results.append(
            {
                "method": name,
                "file_mb": round(sizes[name] / 1024**2, 2),
                "load_seconds_median": round(float(np.median(times)), 3),
                "peak_ram_mb": round(float(np.median(peaks)), 1),
                "frame_mb": round(memory_mb(frame), 1),
                "rows": int(len(frame)),
            }
        )
    config.FORMAT_BENCH_JSON.write_text(json.dumps(results, indent=2))
    return pd.DataFrame(results)


def save_sample(df, n=1000):
    """Small stratified sample committed to the repository for quick tests."""
    parts = []
    for _, group in df.groupby(config.TARGET):
        # Proportional share, but at least 10 rows so rare classes appear.
        size = min(len(group), max(10, int(n * len(group) / len(df))))
        parts.append(group.sample(size, random_state=config.RANDOM_STATE))
    sample = pd.concat(parts, ignore_index=True)
    config.SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample[config.RAW_COLUMNS].to_csv(config.SAMPLE_CSV, index=False)
    return sample


def prepare_data(force=False):
    """Full preparation step used by the menu and by `main.py --all`."""
    download_dataset(force=force)
    df, memory_report = measure_memory()
    df = collapse_one_hot(df)
    save_parquet(df)
    save_sample(df)
    return df, memory_report
