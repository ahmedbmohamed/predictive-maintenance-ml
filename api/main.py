"""REST API that scores machine sensor readings.

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pdm.predictor import Predictor

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("MODEL_PATH")
    state["predictor"] = Predictor.load(Path(model_path)) if model_path else Predictor.load()
    yield
    state.clear()


app = FastAPI(
    title="Predictive Maintenance API",
    description="Failure prediction and anomaly detection for industrial machines.",
    version="1.0.0",
    lifespan=lifespan,
)


class Reading(BaseModel):
    """One snapshot of a machine's sensors. Ranges reject impossible values."""

    product_type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temp_k: float = Field(ge=250, le=350, examples=[298.1])
    process_temp_k: float = Field(ge=250, le=400, examples=[308.6])
    rotational_speed_rpm: float = Field(ge=0, le=5000, examples=[1551])
    torque_nm: float = Field(ge=0, le=150, examples=[42.8])
    tool_wear_min: float = Field(ge=0, le=500, examples=[0])


class Prediction(BaseModel):
    failure_probability: float
    failure_predicted: bool
    anomaly_score: float
    is_anomaly: bool
    status: Literal["OK", "ANOMALY", "FAILURE_RISK"]
    reasons: list[str]


class BatchRequest(BaseModel):
    readings: list[Reading] = Field(min_length=1, max_length=1000)


def _score(readings: list[Reading]) -> list[dict]:
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([r.model_dump() for r in readings])
    return predictor.predict(df)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/model")
def model_info() -> dict:
    return state["predictor"].info


@app.post("/predict", response_model=Prediction)
def predict(reading: Reading) -> dict:
    return _score([reading])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(request: BatchRequest) -> list[dict]:
    return _score(request.readings)
