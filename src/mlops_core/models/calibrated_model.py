"""Modelo calibrado serializable: base LightGBM + calibrador + contrato de features.

Es el artefacto que sirve la API: recibe filas crudas, reconstruye las mismas features
del entrenamiento (con las categorías guardadas) y devuelve una probabilidad **calibrada**
y la decisión al umbral elegido.
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
    base: object  # LGBMClassifier entrenado
    calibrator: Calibrator
    config: DomainConfig
    categories: dict[str, list]
    threshold: float

    def predict_proba(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Probabilidad calibrada de la clase positiva para filas crudas."""
        X, _ = build_features(self.config, raw_df, categories=self.categories)
        p_raw = self.base.predict_proba(X)[:, 1]
        return self.calibrator.transform(p_raw)

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
