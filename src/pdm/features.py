"""Physics-based feature engineering.

The AI4I documentation describes how each failure mode happens. Instead of
letting the model guess these interactions, three features express them
directly:

* heat dissipation failure depends on the gap between process and air
  temperature, combined with a low rotational speed;
* power failure happens when mechanical power (torque x angular speed) is
  too low or too high;
* overstrain failure depends on tool wear multiplied by torque.
"""

import math

import pandas as pd

from pdm.config import PRODUCT_TYPES, SENSOR_COLUMNS

ENGINEERED_COLUMNS = ["temp_diff_k", "power_w", "strain_min_nm"]
TYPE_COLUMNS = [f"type_{t}" for t in PRODUCT_TYPES]
FEATURE_COLUMNS = SENSOR_COLUMNS + ENGINEERED_COLUMNS + TYPE_COLUMNS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    angular_speed = out["rotational_speed_rpm"] * 2 * math.pi / 60
    out["power_w"] = out["torque_nm"] * angular_speed
    out["strain_min_nm"] = out["tool_wear_min"] * out["torque_nm"]
    return out


def encode_product_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the product quality type with a fixed column set,
    so a single prediction gets the same columns as the training data."""
    out = df.copy()
    for t in PRODUCT_TYPES:
        out[f"type_{t}"] = (out["product_type"] == t).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input matrix with columns in a fixed order."""
    return encode_product_type(add_engineered_features(df))[FEATURE_COLUMNS]
