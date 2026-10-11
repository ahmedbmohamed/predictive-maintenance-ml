import math

import numpy as np
import pandas as pd
import pytest

from pdm.config import FAILURE_MODES, TARGET
from pdm.data import DataValidationError, clean, label_consistency_report, load_dataset, load_raw
from pdm.evaluate import best_threshold
from pdm.features import FEATURE_COLUMNS, build_features


def test_dataset_matches_published_statistics():
    df = load_dataset()
    assert len(df) == 10_000
    assert df[TARGET].sum() == 339
    assert set(df["product_type"]) == {"L", "M", "H"}


def test_label_inconsistencies_are_reported():
    report = label_consistency_report(load_dataset())
    assert report == {"failure_without_mode": 9, "mode_without_failure": 18}


def test_failure_modes_never_used_as_features():
    leaking = set(FAILURE_MODES) | {"UDI", "Product ID", TARGET}
    assert not leaking & set(FEATURE_COLUMNS)


def test_missing_column_is_rejected():
    raw = load_raw().drop(columns=["Torque [Nm]"])
    with pytest.raises(DataValidationError, match="Missing columns"):
        clean(raw)


def test_out_of_range_value_is_rejected():
    raw = load_raw()
    raw.loc[0, "Rotational speed [rpm]"] = -10
    with pytest.raises(DataValidationError, match="rotational_speed_rpm"):
        clean(raw)


def test_unknown_product_type_is_rejected():
    raw = load_raw()
    raw.loc[0, "Type"] = "X"
    with pytest.raises(DataValidationError, match="Unknown product types"):
        clean(raw)


def test_engineered_features_follow_physics(normal_reading):
    X = build_features(normal_reading)
    assert list(X.columns) == FEATURE_COLUMNS
    row = X.iloc[0]
    assert row["temp_diff_k"] == pytest.approx(10.5)
    assert row["power_w"] == pytest.approx(42.8 * 1551 * 2 * math.pi / 60)
    assert row["strain_min_nm"] == 0
    assert (row["type_L"], row["type_M"], row["type_H"]) == (0, 1, 0)


def test_one_hot_columns_exist_for_a_single_reading():
    one = pd.DataFrame(
        [
            {
                "product_type": "H",
                "air_temp_k": 300,
                "process_temp_k": 310,
                "rotational_speed_rpm": 1500,
                "torque_nm": 40,
                "tool_wear_min": 10,
            }
        ]
    )
    assert build_features(one).shape == (1, len(FEATURE_COLUMNS))


def test_best_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_threshold(y, scores, beta=2)
    assert 0.3 < t <= 0.8
