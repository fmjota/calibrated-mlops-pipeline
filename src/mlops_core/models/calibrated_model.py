"""Modelo calibrado serializable: base (LightGBM o Bayesiano) + contrato de features.

Es el artefacto que sirve la API: recibe filas crudas, reconstruye las mismas features
del entrenamiento y devuelve una probabilidad (calibrada o posterior bayesiana) y la
decisión al umbral. Para el dominio bayesiano, expone además intervalos creíbles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlops_core.config import DomainConfig
from mlops_core.features import build_features
from mlops_core.models.calibrate import Calibrator


@dataclass
class CalibratedModel:
    base: object  # LGBMClassifier o BayesianLogisticModel entrenado
    calibrator: Calibrator | None  # None para el modelo bayesiano
    config: DomainConfig
    categories: dict[str, list]
    threshold: float

    @property
    def is_bayesian(self) -> bool:
        return self.config.model.type == "bayesian"

    def predict_proba(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Probabilidad de la clase positiva para filas crudas.

        Args:
            raw_df: DataFrame con las columnas del dominio (sin feature engineering).

        Returns:
            np.ndarray [n_rows]: probabilidad calibrada (LightGBM) o media de la
            posterior predictiva (bayesiano).
        """
        if self.is_bayesian:
            return self.base.predict_proba(raw_df)
        X, _ = build_features(self.config, raw_df, categories=self.categories)
        p_raw = self.base.predict_proba(X)[:, 1]
        return self.calibrator.transform(p_raw)

    def predict_interval(
        self, raw_df: pd.DataFrame, level: float | None = None
    ) -> np.ndarray | None:
        """Intervalo creíble por fila (solo disponible para el modelo bayesiano).

        Args:
            raw_df: DataFrame con las columnas del dominio.
            level: nivel del intervalo (usa `config.evaluation.ci_level` si None).

        Returns:
            np.ndarray [n_rows, 2] con (ci_low, ci_high), o None si no es bayesiano.
        """
        if not self.is_bayesian:
            return None
        level = level or self.config.evaluation.ci_level
        return self.base.predict_interval(raw_df, level=level)

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Decisión binaria al umbral de bloqueo/intervención."""
        return (self.predict_proba(raw_df) >= self.threshold).astype(int)


def save_model(model: CalibratedModel, artifacts_dir: str = "artifacts") -> str:
    """Serializa el modelo en artifacts/<domain>/model.joblib y devuelve la ruta.

    Args:
        model: modelo calibrado a persistir.
        artifacts_dir: carpeta base; se crea `<artifacts_dir>/<domain>/`.

    Returns:
        str: ruta del archivo `.joblib` escrito.

    Efectos secundarios:
        Crea directorios y escribe un archivo joblib en disco.
    """
    out_dir = Path(artifacts_dir) / model.config.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.joblib"
    joblib.dump(model, path)
    return str(path)


def load_model(path: str) -> CalibratedModel:
    return joblib.load(path)
