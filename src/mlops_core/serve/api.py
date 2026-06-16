"""API de inferencia FastAPI.

Carga el modelo calibrado serializado y expone:
- `GET /health`  — liveness, no requiere el modelo.
- `POST /predict` — devuelve probabilidad **calibrada** + decisión al umbral.

El modelo se resuelve desde `MODEL_PATH` (por defecto `artifacts/<domain>/model.joblib`)
y se cachea tras la primera petición.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from mlops_core.models import load_model
from mlops_core.serve.schemas import PredictionResponse, Transaction

app = FastAPI(title="MLOps Core — Inferencia", version="0.1.0")


@lru_cache(maxsize=1)
def get_model():
    path = os.environ.get("MODEL_PATH", "artifacts/fraud/model.joblib")
    return load_model(path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction) -> PredictionResponse:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Modelo no disponible.") from exc

    df = pd.DataFrame([tx.model_dump()])
    prob = float(model.predict_proba(df)[0])
    ci_low, ci_high = None, None
    if model.is_bayesian:
        intervals = model.predict_interval(df)
        if intervals is not None:
            ci_low, ci_high = float(intervals[0, 0]), float(intervals[0, 1])
    return PredictionResponse(
        probability=prob,
        decision=int(prob >= model.threshold),
        threshold=float(model.threshold),
        domain=model.config.domain,
        ci_low=ci_low,
        ci_high=ci_high,
    )
