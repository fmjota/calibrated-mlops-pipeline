"""Entrenamiento end-to-end: features → split → modelo → calibración → métricas.

Decisiones con sello estadístico:
- **Split temporal** cuando hay columna de tiempo; aleatorio si no.
- **LightGBM** (path por defecto): tabular desbalanceado, costo/explicabilidad.
- **Bayesiano** (`model.type: bayesian`): regresión logística PyMC con intervalos creíbles
  — para dominios donde la incertidumbre es parte del argumento (ej. comité clínico).
- **Calibración** en un set aparte (path LightGBM) o por construcción (path bayesiano).
- Métricas honestas (PR-AUC, Brier) en ambos paths.
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
    # Filtra claves que son solo del path bayesiano para no confundir a LightGBM.
    bayesian_keys = {"draws", "tune", "chains", "target_accept", "sample_size"}
    params = {"random_state": 42, "n_jobs": -1, "verbosity": -1}
    params.update({k: v for k, v in cfg.model.params.items() if k not in bayesian_keys})
    return params


def _time_sorted(cfg: DomainConfig, df: pd.DataFrame) -> pd.DataFrame:
    col = cfg.model.time_split_column or cfg.columns.datetime
    if col and col in df.columns:
        return df.sort_values(col).reset_index(drop=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _interval_coverage(model: CalibratedModel, X_raw: pd.DataFrame, y: pd.Series) -> float:
    """Fracción de filas cuya probabilidad cae dentro del intervalo creíble."""
    intervals = model.predict_interval(X_raw)
    if intervals is None:
        return float("nan")
    p = model.predict_proba(X_raw)
    inside = (p >= intervals[:, 0]) & (p <= intervals[:, 1])
    return float(inside.mean())


def train_model(
    cfg: DomainConfig,
    df: pd.DataFrame,
    *,
    artifacts_dir: str = "artifacts",
    log_to_mlflow: bool = True,
) -> TrainResult:
    """Entrena, calibra, evalúa, persiste el modelo y (opcional) registra en MLflow.

    Ramifica por `cfg.model.type`:
    - `lightgbm` → LGBMClassifier + calibración isotónica/Platt.
    - `bayesian`  → BayesianLogisticModel (PyMC) + sin calibración extra.

    Args:
        cfg: config del dominio (columnas, params del modelo, calibración, umbral).
        df: datos crudos validados (pandas), con la columna target.
        artifacts_dir: carpeta base donde se guarda `<domain>/model.joblib`.
        log_to_mlflow: si True, registra params, métricas y el artefacto en MLflow.

    Returns:
        TrainResult con modelo, métricas, umbral, método de calibración y ruta.

    Efectos secundarios:
        Escribe el modelo en disco; si `log_to_mlflow`, crea corrida MLflow.
    """
    if cfg.model.type == "bayesian":
        return _train_bayesian(cfg, df, artifacts_dir=artifacts_dir, log_to_mlflow=log_to_mlflow)
    return _train_lgbm(cfg, df, artifacts_dir=artifacts_dir, log_to_mlflow=log_to_mlflow)


# ── path LightGBM ────────────────────────────────────────────────────────────


def _train_lgbm(
    cfg: DomainConfig,
    df: pd.DataFrame,
    *,
    artifacts_dir: str,
    log_to_mlflow: bool,
) -> TrainResult:
    df = _time_sorted(cfg, df)
    X, y = build_features(cfg, df)
    categories = extract_categories(X)

    n = len(X)
    i_train, i_cal = int(n * 0.70), int(n * 0.85)
    X_tr, y_tr = X.iloc[:i_train], y.iloc[:i_train]
    X_ca, y_ca = X.iloc[i_train:i_cal], y.iloc[i_train:i_cal]
    X_te, y_te = X.iloc[i_cal:], y.iloc[i_cal:]

    params = _lgbm_params(cfg)
    base = LGBMClassifier(**params)
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
        _log_to_mlflow(cfg, params, method, threshold, metrics, model_path)

    return TrainResult(
        model=model,
        metrics=metrics,
        threshold=threshold,
        calibration_method=method,
        model_path=model_path,
    )


# ── path Bayesiano ───────────────────────────────────────────────────────────


def _train_bayesian(
    cfg: DomainConfig,
    df: pd.DataFrame,
    *,
    artifacts_dir: str,
    log_to_mlflow: bool,
) -> TrainResult:
    from mlops_core.models.bayesian import BayesianLogisticModel

    df = _time_sorted(cfg, df)

    n = len(df)
    i_te = int(n * (1.0 - cfg.model.valid_fraction))
    df_tr, df_te = df.iloc[:i_te].reset_index(drop=True), df.iloc[i_te:].reset_index(drop=True)

    # El modelo bayesiano trabaja directamente sobre el DataFrame (hace su propio preproceso).
    y_tr = df_tr[cfg.target].astype(int)
    y_te = df_te[cfg.target].astype(int)

    params = cfg.model.params
    base = BayesianLogisticModel(
        numeric_cols=cfg.columns.numeric,
        categorical_cols=cfg.columns.categorical,
        ci_level=cfg.evaluation.ci_level,
    )
    base.fit(
        df_tr,
        y_tr,
        draws=int(params.get("draws", 500)),
        tune=int(params.get("tune", 200)),
        chains=int(params.get("chains", 2)),
        target_accept=float(params.get("target_accept", 0.9)),
        sample_size=int(params.get("sample_size", 10_000)),
    )

    p_te = base.predict_proba(df_te)
    threshold = threshold_for_precision(y_te, p_te, cfg.evaluation.target_precision)
    at_thr = metrics_at_threshold(y_te, p_te, threshold)

    model = CalibratedModel(
        base=base,
        calibrator=None,
        config=cfg,
        categories={},
        threshold=threshold,
    )
    coverage = _interval_coverage(model, df_te, y_te)

    metrics = {
        "pr_auc": pr_auc(y_te, p_te),
        "brier": brier(y_te, p_te),
        "roc_auc": roc_auc(y_te, p_te),
        "ci_coverage": coverage,
        "precision_at_threshold": at_thr["precision"],
        "recall_at_threshold": at_thr["recall"],
        "f1_at_threshold": at_thr["f1"],
        "n_test": float(len(y_te)),
        "positives_test": float(int(y_te.sum())),
    }

    model_path = save_model(model, artifacts_dir)
    method = "bayesian_posterior"

    if log_to_mlflow:
        _log_to_mlflow(cfg, params, method, threshold, metrics, model_path)

    return TrainResult(
        model=model,
        metrics=metrics,
        threshold=threshold,
        calibration_method=method,
        model_path=model_path,
    )


# ── MLflow ────────────────────────────────────────────────────────────────────


def _log_to_mlflow(cfg, params, method, threshold, metrics, model_path) -> None:
    import mlflow

    mlflow.set_experiment(f"proyecto1-{cfg.domain}")
    with mlflow.start_run():
        mlflow.log_params(
            {
                **{str(k): v for k, v in params.items()},
                "calibration_method": method,
                "domain": cfg.domain,
                "threshold": threshold,
            }
        )
        mlflow.log_metrics(
            {
                k: v
                for k, v in metrics.items()
                if not isinstance(v, float) or not __import__("math").isnan(v)
            }
        )
        mlflow.log_artifact(model_path)
