"""Central configuration: paths, column names and training settings."""

import os
from pathlib import Path

# Project root. PDM_HOME wins when set (the Docker image sets it). Otherwise
# use the source tree for an editable install, or the working directory.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get("PDM_HOME") or (_SOURCE_ROOT if (_SOURCE_ROOT / "data").is_dir() else Path.cwd())
)
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw sensor columns renamed to short, code-friendly names.
RENAME = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Type": "product_type",
    "Machine failure": "machine_failure",
}

SENSOR_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
PRODUCT_TYPES = ["L", "M", "H"]
TARGET = "machine_failure"

# Failure mode labels. They describe *why* a machine failed, so they are
# only known after the failure and must never be used as model inputs.
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# A missed failure (unplanned downtime) costs much more than a false alarm
# (one extra inspection), so the decision threshold is tuned on F2, which
# weights recall twice as much as precision.
F_BETA = 2.0
