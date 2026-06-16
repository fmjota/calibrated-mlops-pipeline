"""La API responde health, predice con datos válidos y rechaza datos inválidos."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlops_core.config import load_config
from mlops_core.models import train_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_synthetic import generate  # noqa: E402

VALID_TX = {
    "trans_date_trans_time": "2020-06-01T12:30:00",
    "amt": 125.50,
    "category": "grocery_pos",
    "gender": "F",
    "state": "CA",
    "city_pop": 50000,
    "lat": 37.77,
    "long": -122.41,
    "merch_lat": 37.80,
    "merch_long": -122.30,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs" / "fraud.yaml")
    cfg.model.params["n_estimators"] = 60
    df = generate(rows=3000, seed=2, fraud_rate=0.12)
    result = train_model(cfg, df, artifacts_dir=str(tmp_path), log_to_mlflow=False)

    monkeypatch.setenv("MODEL_PATH", result.model_path)
    from mlops_core.serve import api

    api.get_model.cache_clear()
    yield TestClient(api.app)
    api.get_model.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_TX)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in (0, 1)
    assert body["domain"] == "fraud"


def test_predict_rejects_negative_amount(client):
    bad = {**VALID_TX, "amt": -10}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_unknown_category(client):
    bad = {**VALID_TX, "category": "crypto"}
    assert client.post("/predict", json=bad).status_code == 422
