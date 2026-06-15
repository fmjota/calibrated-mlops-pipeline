"""El feature engineering es consistente entre train e inferencia."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops_core.config import load_config
from mlops_core.features import build_features, extract_categories

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg():
    return load_config(CONFIGS / "fraud.yaml")


def test_engineered_columns_present(valid_fraud_df):
    X, y = build_features(_cfg(), valid_fraud_df)
    for col in ("hour", "dayofweek", "month", "geo_distance_km"):
        assert col in X.columns
    assert (X["geo_distance_km"] >= 0).all()
    assert y is not None and set(y.unique()).issubset({0, 1})


def test_categorical_dtype(valid_fraud_df):
    X, _ = build_features(_cfg(), valid_fraud_df)
    assert isinstance(X["category"].dtype, pd.CategoricalDtype)
    cats = extract_categories(X)
    assert "category" in cats and "gender" in cats and "state" in cats


def test_inference_without_target(valid_fraud_df):
    df = valid_fraud_df.drop(columns=["is_fraud"])
    X, y = build_features(_cfg(), df)
    assert y is None
    assert "amt" in X.columns


def test_categories_enforced_for_consistency(valid_fraud_df):
    X_train, _ = build_features(_cfg(), valid_fraud_df)
    cats = extract_categories(X_train)
    # Una fila con categoría no vista en train debe quedar como NaN (código consistente).
    new = valid_fraud_df.iloc[[0]].copy()
    new.loc[new.index[0], "category"] = "categoria_nueva"
    X_inf, _ = build_features(_cfg(), new, categories=cats)
    assert X_inf["category"].isna().all()
    assert list(X_inf["category"].cat.categories) == cats["category"]
