"""Dominio educación: esquema de deserción y generalización del núcleo LightGBM.

Este test demuestra que el mismo núcleo (sin código nuevo de modelo) sirve al dominio
educación. El drift PSI/KS detecta cambios de cohorte año a año.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mlops_core.config import load_config  # noqa: E402
from mlops_core.drift import detect_drift  # noqa: E402
from mlops_core.models import train_model  # noqa: E402
from mlops_core.validate import DataValidationError, validate_dataframe  # noqa: E402

CONFIGS = ROOT / "configs"


@pytest.fixture
def valid_dropout_df() -> pd.DataFrame:
    """DataFrame pequeño válido según el esquema de deserción."""
    n = 80
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Marital Status": rng.integers(1, 4, n),
            "Application mode": rng.integers(1, 18, n),
            "Application order": rng.integers(0, 6, n),
            "Course": rng.integers(1, 20, n),
            "Daytime/evening attendance": rng.integers(0, 2, n),
            "Previous qualification": rng.integers(1, 17, n),
            "Previous qualification (grade)": rng.uniform(100, 190, n),
            "Admission grade": rng.uniform(100, 190, n),
            "Age at enrollment": rng.integers(17, 45, n),
            "Gender": rng.integers(0, 2, n),
            "Scholarship holder": rng.integers(0, 2, n),
            "Debtor": rng.integers(0, 2, n),
            "Tuition fees up to date": rng.integers(0, 2, n),
            "Curricular units 1st sem (approved)": rng.integers(0, 8, n),
            "Curricular units 2nd sem (approved)": rng.integers(0, 8, n),
            "Curricular units 1st sem (grade)": rng.uniform(0, 18, n),
            "Curricular units 2nd sem (grade)": rng.uniform(0, 18, n),
            "Unemployment rate": rng.uniform(5, 20, n),
            "Inflation rate": rng.uniform(-1, 5, n),
            "GDP": rng.uniform(-4, 4, n),
            "dropout": rng.integers(0, 2, n),
        }
    )


# ── schema tests ──────────────────────────────────────────────────────────────


def test_valid_dropout_passes(valid_dropout_df):
    out = validate_dataframe(valid_dropout_df, "dropout")
    assert len(out) == len(valid_dropout_df)


def test_invalid_age_rejected(valid_dropout_df):
    valid_dropout_df.loc[0, "Age at enrollment"] = 5  # < 15
    with pytest.raises(DataValidationError):
        validate_dataframe(valid_dropout_df, "dropout")


def test_invalid_grade_rejected(valid_dropout_df):
    valid_dropout_df.loc[0, "Curricular units 1st sem (grade)"] = 25  # > 20
    with pytest.raises(DataValidationError):
        validate_dataframe(valid_dropout_df, "dropout")


# ── model reuse tests ─────────────────────────────────────────────────────────


def test_lgbm_core_serves_dropout(tmp_path, valid_dropout_df):
    """El mismo núcleo LightGBM funciona con educación sin cambiar código del modelo."""
    cfg = load_config(CONFIGS / "dropout.yaml")
    cfg.model.params["n_estimators"] = 50
    df = pd.concat([valid_dropout_df] * 5, ignore_index=True)
    result = train_model(cfg, df, artifacts_dir=str(tmp_path), log_to_mlflow=False)

    assert result.calibration_method in ("isotonic", "sigmoid")
    probs = result.model.predict_proba(df.head(5))
    assert ((probs >= 0) & (probs <= 1)).all()


def test_drift_detects_cohort_shift(valid_dropout_df):
    """El drift PSI/KS detecta cambios de cohorte: sube el GDP y la tasa de desempleo."""
    cfg = load_config(CONFIGS / "dropout.yaml")
    reference = valid_dropout_df.copy()
    current = valid_dropout_df.copy()
    # Simula un cambio de cohorte: peores condiciones económicas
    current["Unemployment rate"] = current["Unemployment rate"] * 2 + 10
    current["GDP"] = current["GDP"] - 5

    report = detect_drift(reference, current, cfg)
    assert report.drifted
    drifted_cols = report.drifted_features
    assert any(c in drifted_cols for c in ["Unemployment rate", "GDP"])
