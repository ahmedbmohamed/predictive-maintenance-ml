"""Load and validate the AI4I 2020 predictive maintenance dataset."""

from pathlib import Path

import pandas as pd

from pdm.config import (
    DATA_PATH,
    FAILURE_MODES,
    PRODUCT_TYPES,
    RENAME,
    SENSOR_COLUMNS,
    TARGET,
)

# Physically plausible ranges. Values outside them point to a sensor fault
# or a data entry error rather than a real machine state.
VALID_RANGES = {
    "air_temp_k": (250.0, 350.0),
    "process_temp_k": (250.0, 400.0),
    "rotational_speed_rpm": (0.0, 5000.0),
    "torque_nm": (0.0, 150.0),
    "tool_wear_min": (0.0, 500.0),
}


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV (the original file starts with a UTF-8 BOM)."""
    return pd.read_csv(path, encoding="utf-8-sig")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and validate schema, types, ranges and labels."""
    df = df.rename(columns=RENAME)

    required = SENSOR_COLUMNS + ["product_type", TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise DataValidationError("Dataset contains missing values")

    unknown_types = set(df["product_type"]) - set(PRODUCT_TYPES)
    if unknown_types:
        raise DataValidationError(f"Unknown product types: {unknown_types}")

    for col, (low, high) in VALID_RANGES.items():
        bad = ~df[col].between(low, high)
        if bad.any():
            raise DataValidationError(f"{bad.sum()} values of {col} outside [{low}, {high}]")

    if not set(df[TARGET].unique()) <= {0, 1}:
        raise DataValidationError("Target must be binary")

    return df


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def label_consistency_report(df: pd.DataFrame) -> dict:
    """Count rows where the failure flag and the failure mode labels disagree.

    The public dataset has a few such rows (for example random failures
    without the machine failure flag). They are kept as they are, and the
    count is reported so the choice is transparent.
    """
    modes = [m for m in FAILURE_MODES if m in df.columns]
    any_mode = df[modes].sum(axis=1) > 0
    return {
        "failure_without_mode": int(((df[TARGET] == 1) & ~any_mode).sum()),
        "mode_without_failure": int(((df[TARGET] == 0) & any_mode).sum()),
    }
