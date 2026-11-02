"""Project-wide constants: paths, column names, lookup tables and seeds.

Every other module imports from here, so a path or a column name is
defined in exactly one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

DATA_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
DATA_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/31/covertype"
RAW_ZIP = RAW_DIR / "covertype.zip"
RAW_GZ = RAW_DIR / "covtype.data.gz"
RAW_CSV = RAW_DIR / "covtype.csv"
PARQUET_PATH = PROCESSED_DIR / "covtype.parquet"
SAMPLE_CSV = DATA_DIR / "sample" / "covtype_sample_1000.csv"
DB_PATH = PROCESSED_DIR / "covtype.db"

AUDIT_JSON = REPORTS_DIR / "data_audit.json"
MEMORY_JSON = REPORTS_DIR / "memory_report.json"
FORMAT_BENCH_JSON = REPORTS_DIR / "format_benchmark.json"
RESULTS_PARQUET = REPORTS_DIR / "experiments.parquet"
RESULTS_CSV = REPORTS_DIR / "experiments.csv"
TUNING_JSON = REPORTS_DIR / "tuning_results.json"
SQL_RESULTS_DIR = REPORTS_DIR / "sql"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_META = MODELS_DIR / "best_model.json"
HISTORY_JSON = REPORTS_DIR / "prediction_history.json"

# ---------------------------------------------------------------------------
# Columns (the raw file has no header, the order comes from covtype.info)
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]
TARGET = "Cover_Type"
RAW_COLUMNS = NUMERIC_COLS + WILDERNESS_COLS + SOIL_COLS + [TARGET]

# Compact categorical columns built from the one-hot blocks.
WILDERNESS = "Wilderness"
SOIL = "Soil"

# ---------------------------------------------------------------------------
# Lookup tables (from covtype.info)
# ---------------------------------------------------------------------------
COVER_TYPES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

WILDERNESS_AREAS = {
    1: "Rawah",
    2: "Neota",
    3: "Comanche Peak",
    4: "Cache la Poudre",
}

# Study code -> (USFS ELU code, description)
SOIL_TYPES = {
    1: (2702, "Cathedral family - Rock outcrop complex, extremely stony"),
    2: (2703, "Vanet - Ratake families complex, very stony"),
    3: (2704, "Haploborolis - Rock outcrop complex, rubbly"),
    4: (2705, "Ratake family - Rock outcrop complex, rubbly"),
    5: (2706, "Vanet family - Rock outcrop complex, rubbly"),
    6: (2717, "Vanet - Wetmore families - Rock outcrop complex, stony"),
    7: (3501, "Gothic family"),
    8: (3502, "Supervisor - Limber families complex"),
    9: (4201, "Troutville family, very stony"),
    10: (4703, "Bullwark - Catamount families - Rock outcrop complex, rubbly"),
    11: (4704, "Bullwark - Catamount families - Rock land complex, rubbly"),
    12: (4744, "Legault family - Rock land complex, stony"),
    13: (4758, "Catamount family - Rock land - Bullwark family complex, rubbly"),
    14: (5101, "Pachic Argiborolis - Aquolis complex"),
    15: (5151, "unspecified in the USFS Soil and ELU Survey"),
    16: (6101, "Cryaquolis - Cryoborolis complex"),
    17: (6102, "Gateview family - Cryaquolis complex"),
    18: (6731, "Rogert family, very stony"),
    19: (7101, "Typic Cryaquolis - Borohemists complex"),
    20: (7102, "Typic Cryaquepts - Typic Cryaquolls complex"),
    21: (7103, "Typic Cryaquolls - Leighcan family, till substratum complex"),
    22: (7201, "Leighcan family, till substratum, extremely bouldery"),
    23: (7202, "Leighcan family, till substratum - Typic Cryaquolls complex"),
    24: (7700, "Leighcan family, extremely stony"),
    25: (7701, "Leighcan family, warm, extremely stony"),
    26: (7702, "Granile - Catamount families complex, very stony"),
    27: (7709, "Leighcan family, warm - Rock outcrop complex, extremely stony"),
    28: (7710, "Leighcan family - Rock outcrop complex, extremely stony"),
    29: (7745, "Como - Legault families complex, extremely stony"),
    30: (7746, "Como family - Rock land - Legault family complex, extremely stony"),
    31: (7755, "Leighcan - Catamount families complex, extremely stony"),
    32: (7756, "Catamount family - Rock outcrop - Leighcan family complex, extremely stony"),
    33: (7757, "Leighcan - Catamount families - Rock outcrop complex, extremely stony"),
    34: (7790, "Cryorthents - Rock land complex, extremely stony"),
    35: (8703, "Cryumbrepts - Rock outcrop - Cryaquepts complex"),
    36: (8707, "Bross family - Rock land - Cryumbrepts complex, extremely stony"),
    37: (8708, "Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony"),
    38: (8771, "Leighcan - Moran families - Cryaquolls complex, extremely stony"),
    39: (8772, "Moran family - Cryorthents - Leighcan family complex, extremely stony"),
    40: (8776, "Moran family - Cryorthents - Rock land complex, extremely stony"),
}

# First ELU digit -> climatic zone, second digit -> geologic zone.
CLIMATIC_ZONES = {
    1: "lower montane dry",
    2: "lower montane",
    3: "montane dry",
    4: "montane",
    5: "montane dry and montane",
    6: "montane and subalpine",
    7: "subalpine",
    8: "alpine",
}
GEOLOGIC_ZONES = {
    1: "alluvium",
    2: "glacial",
    3: "shale",
    4: "sandstone",
    5: "mixed sedimentary",
    6: "unspecified",
    7: "igneous and metamorphic",
    8: "volcanic",
}

# Valid physical ranges used by the data audit (from covtype.info).
VALID_RANGES = {
    "Aspect": (0, 360),
    "Slope": (0, 90),
    "Hillshade_9am": (0, 255),
    "Hillshade_Noon": (0, 255),
    "Hillshade_3pm": (0, 255),
    "Horizontal_Distance_To_Hydrology": (0, None),
    "Horizontal_Distance_To_Roadways": (0, None),
    "Horizontal_Distance_To_Fire_Points": (0, None),
}

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CHUNK_SIZE = 100_000
TUNING_SAMPLE = 40_000
CV_FOLDS = 3


def ensure_dirs():
    """Create every output directory the pipeline writes to."""
    for folder in (
        RAW_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        SQL_RESULTS_DIR,
        SAMPLE_CSV.parent,
    ):
        folder.mkdir(parents=True, exist_ok=True)
