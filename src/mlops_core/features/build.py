"""Construcción de features dirigida por config (misma lógica en train y en serving).

Es clave que train e inferencia usen exactamente la misma transformación. Por eso el
modelo serializado guarda las categorías vistas en entrenamiento y se reaplican aquí
(`categories=`), para que los códigos de las columnas categóricas sean consistentes y
LightGBM no se confunda con una categoría nueva.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_core.config import DomainConfig

ENGINEERED = ["hour", "dayofweek", "month", "geo_distance_km"]
_GEO_COLS = {"lat", "long", "merch_lat", "merch_long"}


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia en km entre dos puntos (cardholder y comercio)."""
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def build_features(
    cfg: DomainConfig,
    df: pd.DataFrame,
    categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Devuelve (X, y). `y` es None si la columna target no está en `df` (inferencia)."""
    df = df.copy()

    dtc = cfg.columns.datetime
    if dtc and dtc in df.columns:
        ts = pd.to_datetime(df[dtc])
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["month"] = ts.dt.month

    if _GEO_COLS.issubset(df.columns):
        df["geo_distance_km"] = _haversine_km(
            df["lat"], df["long"], df["merch_lat"], df["merch_long"]
        )

    numeric = [c for c in cfg.columns.numeric if c in df.columns]
    engineered = [c for c in ENGINEERED if c in df.columns]
    categorical = [c for c in cfg.columns.categorical if c in df.columns]
    feature_cols = list(dict.fromkeys([*numeric, *engineered, *categorical]))

    X = df[feature_cols].copy()
    for col in categorical:
        if categories is not None and col in categories:
            X[col] = pd.Categorical(X[col], categories=categories[col])
        else:
            X[col] = X[col].astype("category")

    y = df[cfg.target].astype(int) if cfg.target in df.columns else None
    return X, y


def extract_categories(X: pd.DataFrame) -> dict[str, list]:
    """Categorías vistas por columna categórica, para reaplicar en inferencia."""
    return {
        col: list(X[col].cat.categories)
        for col in X.columns
        if isinstance(X[col].dtype, pd.CategoricalDtype)
    }
