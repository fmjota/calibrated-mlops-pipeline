"""Detección de data drift por feature: PSI y test KS.

El drift es, en el fondo, **detección de señales aplicada al modelo**: la distribución de
producción se aleja de la de entrenamiento y el modelo degrada en silencio. Lo medimos con
dos lentes complementarios:
- **PSI** (Population Stability Index): magnitud del corrimiento de la distribución.
  Regla habitual: <0.1 estable, 0.1-0.2 leve, >0.2 relevante.
- **KS** (Kolmogorov-Smirnov): test de que ambas muestras vienen de la misma distribución.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from mlops_core.config import DomainConfig


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index entre una muestra de referencia y una actual.

    Usa cuantiles de la referencia como bordes (con colas abiertas) y compara las
    proporciones por bin. Mayor PSI = mayor corrimiento.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:  # feature constante en la referencia
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_test(expected, actual) -> tuple[float, float]:
    """Devuelve (estadístico KS, p-valor) entre dos muestras."""
    result = ks_2samp(np.asarray(expected, dtype=float), np.asarray(actual, dtype=float))
    return float(result.statistic), float(result.pvalue)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    drifted: bool


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    psi_threshold: float
    ks_pvalue_threshold: float

    @property
    def drifted(self) -> bool:
        return any(f.drifted for f in self.features)

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.features])

    def summary(self) -> str:
        if not self.drifted:
            return f"Sin drift relevante en {len(self.features)} features."
        cols = ", ".join(self.drifted_features)
        n_drift, n_total = len(self.drifted_features), len(self.features)
        return f"DRIFT detectado en {n_drift}/{n_total} features: {cols}."


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: DomainConfig,
) -> DriftReport:
    """Compara features numéricas de `current` contra `reference` según los umbrales del config."""
    psi_thr = cfg.drift.psi_threshold
    ks_thr = cfg.drift.ks_pvalue_threshold

    results: list[FeatureDrift] = []
    for col in cfg.columns.numeric:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = reference[col].dropna()
        cur = current[col].dropna()
        col_psi = psi(ref, cur)
        ks_stat, ks_p = ks_test(ref, cur)
        drifted = col_psi > psi_thr or ks_p < ks_thr
        results.append(FeatureDrift(col, col_psi, ks_stat, ks_p, drifted))

    return DriftReport(results, psi_thr, ks_thr)
