"""Entrenamiento end-to-end: features -> split temporal -> LightGBM -> calibración -> métricas.

Decisiones con sello estadístico:
- **Split temporal** (no aleatorio): entrenar en el pasado y evaluar en el futuro evita
  fuga temporal y refleja cómo opera el modelo en producción.
- **LightGBM con `is_unbalance`**: en tabular desbalanceado suele ganar a deep learning
  en performance, costo y explicabilidad.
- **Calibración** en un set aparte y métricas honestas (PR-AUC, Brier) en test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lightgbm import LGBMClassifier

from mlops_core.config import DomainConfig
from mlops_core.evaluate.metrics import (
    brier,
    metrics_at_threshold,
    pr_auc,
    roc_auc,
    threshold_for_precision,
)
from mlops_core.features import build_features, extract_categories
from mlops_core.models.calibrate import fit_best_calibrator
from mlops_core.models.calibrated_model import CalibratedModel, save_model


@dataclass
class TrainResult:
    model: CalibratedModel
    metrics: dict[str, float]
    threshold: float
    calibration_method: str
    model_path: str | None = None


def _lgbm_params(cfg: DomainConfig) -> dict:
    params = {"random_state": 42, "n_jobs": -1, "verbosity": -1}
    params.update(cfg.model.params)
    return params


def _time_sorted(cfg: DomainConfig, df: pd.DataFrame) -> pd.DataFrame:
    col = cfg.model.time_split_column or cfg.columns.datetime
    if col and col in df.columns:
        return df.sort_values(col).reset_index(drop=True)
    return df.reset_index(drop=True)


def train_model(
    cfg: DomainConfig,
    df: pd.DataFrame,
    *,
    artifacts_dir: str = "artifacts",
    log_to_mlflow: bool = True,
) -> TrainResult:
    """Entrena, calibra, evalúa, persiste el modelo y (opcional) registra en MLflow.

    Args:
        cfg: config del dominio (columnas, params del modelo, calibración, umbral).
        df: datos crudos validados (pandas), con la columna target.
        artifacts_dir: carpeta base donde se guarda `<domain>/model.joblib`.
        log_to_mlflow: si True, registra params, métricas y el artefacto en MLflow.

    Returns:
        TrainResult: modelo calibrado, métricas (PR-AUC/Brier/ROC-AUC), umbral,
        método de calibración elegido y ruta del modelo guardado.

    Efectos secundarios:
        Escribe el modelo serializado en disco y, si `log_to_mlflow`, crea una corrida
        MLflow (carpeta `mlruns/`).
    """
    df = _time_sorted(cfg, df)
    X, y = build_features(cfg, df)
    categories = extract_categories(X)

    n = len(X)
    i_train, i_cal = int(n * 0.70), int(n * 0.85)
    X_tr, y_tr = X.iloc[:i_train], y.iloc[:i_train]
    X_ca, y_ca = X.iloc[i_train:i_cal], y.iloc[i_train:i_cal]
    X_te, y_te = X.iloc[i_cal:], y.iloc[i_cal:]

    base = LGBMClassifier(**_lgbm_params(cfg))
    base.fit(X_tr, y_tr)

    p_ca = base.predict_proba(X_ca)[:, 1]
    p_te_raw = base.predict_proba(X_te)[:, 1]

    calibrator, method = fit_best_calibrator(
        p_ca, y_ca, p_te_raw, y_te, method=cfg.calibration.method
    )
    p_te_cal = calibrator.transform(p_te_raw)

    threshold = threshold_for_precision(y_te, p_te_cal, cfg.evaluation.target_precision)
    at_thr = metrics_at_threshold(y_te, p_te_cal, threshold)

    metrics = {
        "pr_auc_raw": pr_auc(y_te, p_te_raw),
        "pr_auc_cal": pr_auc(y_te, p_te_cal),
        "brier_raw": brier(y_te, p_te_raw),
        "brier_cal": brier(y_te, p_te_cal),
        "roc_auc": roc_auc(y_te, p_te_raw),
        "precision_at_threshold": at_thr["precision"],
        "recall_at_threshold": at_thr["recall"],
        "f1_at_threshold": at_thr["f1"],
        "n_test": float(len(y_te)),
        "positives_test": float(int(y_te.sum())),
    }

    model = CalibratedModel(
        base=base,
        calibrator=calibrator,
        config=cfg,
        categories=categories,
        threshold=threshold,
    )
    model_path = save_model(model, artifacts_dir)

    if log_to_mlflow:
        _log_to_mlflow(cfg, _lgbm_params(cfg), method, threshold, metrics, model_path)

    return TrainResult(
        model=model,
        metrics=metrics,
        threshold=threshold,
        calibration_method=method,
        model_path=model_path,
    )


def _log_to_mlflow(cfg, params, method, threshold, metrics, model_path) -> None:
    import mlflow

    mlflow.set_experiment(f"proyecto1-{cfg.domain}")
    with mlflow.start_run():
        mlflow.log_params(
            {**params, "calibration_method": method, "domain": cfg.domain, "threshold": threshold}
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_path)
