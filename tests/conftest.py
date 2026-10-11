import pandas as pd
import pytest

from pdm import train


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    """Train a small, fast model in a temporary folder (real data, fewer trees)."""
    out = tmp_path_factory.mktemp("artifacts")
    mp = pytest.MonkeyPatch()
    mp.setattr(train, "MODELS_DIR", out / "models")
    mp.setattr(train, "REPORTS_DIR", out / "reports")
    mp.setattr(train, "FIGURES_DIR", out / "reports" / "figures")
    report = train.main(n_estimators=60, cv_folds=3, make_figures=False)
    mp.undo()
    return {"model_path": out / "models" / "model.joblib", "report": report}


@pytest.fixture
def normal_reading() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_type": "M",
                "air_temp_k": 298.1,
                "process_temp_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 0,
            }
        ]
    )


@pytest.fixture
def overstrained_reading() -> pd.DataFrame:
    # Worn tool under high torque on a low quality product: wear x torque
    # is far above the 11000 overstrain limit for type L.
    return pd.DataFrame(
        [
            {
                "product_type": "L",
                "air_temp_k": 300.0,
                "process_temp_k": 310.0,
                "rotational_speed_rpm": 1300,
                "torque_nm": 65.0,
                "tool_wear_min": 220,
            }
        ]
    )
