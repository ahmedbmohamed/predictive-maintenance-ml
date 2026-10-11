import pytest
from fastapi.testclient import TestClient

from api.main import app

NORMAL = {
    "product_type": "M",
    "air_temp_k": 298.1,
    "process_temp_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}
OVERSTRAINED = {
    "product_type": "L",
    "air_temp_k": 300.0,
    "process_temp_k": 310.0,
    "rotational_speed_rpm": 1300,
    "torque_nm": 65.0,
    "tool_wear_min": 220,
}


@pytest.fixture(scope="module")
def client(trained_bundle):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(trained_bundle["model_path"]))
    with TestClient(app) as c:
        yield c
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model").json()
    assert {"model", "version", "threshold", "test_metrics"} <= set(info)


def test_predict_normal_machine(client):
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    assert r.json()["status"] == "OK"


def test_predict_overstrained_machine(client):
    body = client.post("/predict", json=OVERSTRAINED).json()
    assert body["status"] == "FAILURE_RISK"
    assert body["reasons"]


def test_batch(client):
    r = client.post("/predict/batch", json={"readings": [NORMAL, OVERSTRAINED]})
    assert r.status_code == 200
    assert [p["failure_predicted"] for p in r.json()] == [False, True]


@pytest.mark.parametrize(
    "field,value",
    [("torque_nm", -5), ("rotational_speed_rpm", 99999), ("product_type", "X")],
)
def test_invalid_input_is_rejected(client, field, value):
    assert client.post("/predict", json={**NORMAL, field: value}).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"readings": []}).status_code == 422
