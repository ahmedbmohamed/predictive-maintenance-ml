"""Load the trained bundle and score machine readings.

Used by both the API and the dashboard, so they always share exactly the
same preprocessing, threshold and model version.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from pdm.config import MODELS_DIR
from pdm.features import add_engineered_features, build_features

# Operating limits published with the AI4I 2020 dataset. They are used to
# explain an alert in plain words, not to make the prediction itself.
OVERSTRAIN_LIMIT = {"L": 11_000, "M": 12_000, "H": 13_000}
POWER_RANGE_W = (3_500, 9_000)
HEAT_TEMP_DIFF_K = 8.6
HEAT_SPEED_RPM = 1_380
TOOL_WEAR_WARNING_MIN = 200


def explain(row: pd.Series) -> list[str]:
    """Return the operating limits a reading is close to or beyond."""
    reasons = []
    if row["temp_diff_k"] < HEAT_TEMP_DIFF_K and row["rotational_speed_rpm"] < HEAT_SPEED_RPM:
        reasons.append(
            f"Heat dissipation risk: temperature gap {row['temp_diff_k']:.1f} K "
            f"with speed {row['rotational_speed_rpm']:.0f} rpm"
        )
    low, high = POWER_RANGE_W
    if not low <= row["power_w"] <= high:
        reasons.append(f"Power out of range: {row['power_w']:.0f} W (normal {low} to {high} W)")
    limit = OVERSTRAIN_LIMIT[row["product_type"]]
    if row["strain_min_nm"] > 0.9 * limit:
        reasons.append(
            f"Overstrain risk: wear x torque {row['strain_min_nm']:.0f} "
            f"(limit {limit} for type {row['product_type']})"
        )
    if row["tool_wear_min"] >= TOOL_WEAR_WARNING_MIN:
        reasons.append(f"Tool near end of life: {row['tool_wear_min']:.0f} min of wear")
    return reasons


@dataclass
class Predictor:
    bundle: dict

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "model.joblib") -> "Predictor":
        if not Path(path).exists():
            raise FileNotFoundError(f"No trained model at {path}. Run `python -m pdm.train` first.")
        return cls(joblib.load(path))

    @property
    def info(self) -> dict:
        b = self.bundle
        return {
            "model": b["model_name"],
            "version": b["version"],
            "trained_at": b["trained_at"],
            "threshold": round(b["threshold"], 4),
            "test_metrics": b["test_metrics"],
        }

    def predict(self, readings: pd.DataFrame) -> list[dict]:
        X = build_features(readings)[self.bundle["feature_columns"]]
        proba = self.bundle["model"].predict_proba(X)[:, 1]
        anomaly = -self.bundle["anomaly_detector"].score_samples(X)
        enriched = add_engineered_features(readings)

        results = []
        for i in range(len(readings)):
            failure = bool(proba[i] >= self.bundle["threshold"])
            is_anomaly = bool(anomaly[i] >= self.bundle["anomaly_threshold"])
            if failure:
                status = "FAILURE_RISK"
            elif is_anomaly:
                status = "ANOMALY"
            else:
                status = "OK"
            results.append(
                {
                    "failure_probability": round(float(proba[i]), 4),
                    "failure_predicted": failure,
                    "anomaly_score": round(float(anomaly[i]), 4),
                    "is_anomaly": is_anomaly,
                    "status": status,
                    "reasons": explain(enriched.iloc[i]),
                }
            )
        return results
