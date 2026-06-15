"""Entrenamiento end-to-end sobre datos sintéticos: entrena, calibra, persiste y predice."""

from __future__ import annotations

import sys
from pathlib import Path

from mlops_core.config import load_config
from mlops_core.models import load_model, train_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_synthetic import generate  # noqa: E402


def _cfg_fast():
    cfg = load_config(ROOT / "configs" / "fraud.yaml")
    cfg.model.params["n_estimators"] = 80  # más rápido en tests
    return cfg


def test_train_and_predict(tmp_path):
    cfg = _cfg_fast()
    df = generate(rows=4000, seed=11, fraud_rate=0.12)

    result = train_model(cfg, df, artifacts_dir=str(tmp_path), log_to_mlflow=False)

    # El modelo aprende la señal sintética.
    assert result.metrics["pr_auc_cal"] > 0.5
    assert 0.0 <= result.threshold <= 1.0
    assert result.calibration_method in ("isotonic", "sigmoid")

    # Probabilidades calibradas en rango para filas crudas.
    probs = result.model.predict_proba(df.head(20))
    assert ((probs >= 0) & (probs <= 1)).all()
    preds = result.model.predict(df.head(20))
    assert set(map(int, preds)).issubset({0, 1})


def test_saved_model_roundtrip(tmp_path):
    cfg = _cfg_fast()
    df = generate(rows=3000, seed=5, fraud_rate=0.12)
    result = train_model(cfg, df, artifacts_dir=str(tmp_path), log_to_mlflow=False)

    reloaded = load_model(result.model_path)
    p1 = result.model.predict_proba(df.head(10))
    p2 = reloaded.predict_proba(df.head(10))
    assert (abs(p1 - p2) < 1e-9).all()
