"""Dominio salud: esquema de reingreso, modelo bayesiano con intervalos creíbles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mlops_core.config import load_config  # noqa: E402
from mlops_core.models import train_model  # noqa: E402
from mlops_core.validate import DataValidationError, validate_dataframe  # noqa: E402

CONFIGS = ROOT / "configs"


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def valid_readmission_df() -> pd.DataFrame:
    """DataFrame pequeño válido según el esquema de reingreso."""
    n = 60
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "race": rng.choice(["Caucasian", "AfricanAmerican", "Other"], n),
            "gender": rng.choice(["Female", "Male"], n),
            "age": rng.choice(["[50-60)", "[60-70)", "[70-80)"], n),
            "time_in_hospital": rng.integers(1, 10, n),
            "num_lab_procedures": rng.integers(1, 60, n),
            "num_procedures": rng.integers(0, 5, n),
            "num_medications": rng.integers(1, 30, n),
            "number_outpatient": rng.integers(0, 5, n),
            "number_emergency": rng.integers(0, 5, n),
            "number_inpatient": rng.integers(0, 5, n),
            "number_diagnoses": rng.integers(1, 9, n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["Yes", "No"], n),
            "metformin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "repaglinide": ["No"] * n,
            "nateglinide": ["No"] * n,
            "chlorpropamide": ["No"] * n,
            "glimepiride": ["No"] * n,
            "glipizide": ["No"] * n,
            "glyburide": ["No"] * n,
            "pioglitazone": ["No"] * n,
            "rosiglitazone": ["No"] * n,
            "acarbose": ["No"] * n,
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "readmitted_30d": rng.integers(0, 2, n),
        }
    )


# ── schema tests ──────────────────────────────────────────────────────────────


def test_valid_readmission_passes(valid_readmission_df):
    out = validate_dataframe(valid_readmission_df, "readmission")
    assert len(out) == len(valid_readmission_df)


def test_invalid_gender_rejected(valid_readmission_df):
    valid_readmission_df.loc[0, "gender"] = "Other"
    with pytest.raises(DataValidationError) as exc:
        validate_dataframe(valid_readmission_df, "readmission")
    assert "gender" in exc.value.report.summary()


def test_invalid_time_in_hospital_rejected(valid_readmission_df):
    valid_readmission_df.loc[0, "time_in_hospital"] = 0  # < 1
    with pytest.raises(DataValidationError):
        validate_dataframe(valid_readmission_df, "readmission")


def test_invalid_label_rejected(valid_readmission_df):
    valid_readmission_df.loc[0, "readmitted_30d"] = 2
    with pytest.raises(DataValidationError) as exc:
        validate_dataframe(valid_readmission_df, "readmission")
    assert "readmitted_30d" in exc.value.report.summary()


# ── bayesian model tests ──────────────────────────────────────────────────────


def _small_cfg(tmp_path):
    cfg = load_config(CONFIGS / "readmission.yaml")
    cfg.model.params["draws"] = 50
    cfg.model.params["tune"] = 30
    cfg.model.params["chains"] = 1
    cfg.model.params["sample_size"] = 200
    return cfg


def test_bayesian_train_and_predict(tmp_path, valid_readmission_df):
    cfg = _small_cfg(tmp_path)
    # Forzar señal para que el modelo pueda aprender algo
    df = pd.concat([valid_readmission_df] * 5, ignore_index=True)
    result = train_model(cfg, df, artifacts_dir=str(tmp_path), log_to_mlflow=False)

    assert result.calibration_method == "bayesian_posterior"
    probs = result.model.predict_proba(df.head(5))
    assert ((probs >= 0) & (probs <= 1)).all()
    preds = result.model.predict(df.head(5))
    assert set(map(int, preds)).issubset({0, 1})


def test_bayesian_intervals_have_valid_shape(tmp_path, valid_readmission_df):
    cfg = _small_cfg(tmp_path)
    df = pd.concat([valid_readmission_df] * 5, ignore_index=True)
    result = train_model(cfg, df, artifacts_dir=str(tmp_path), log_to_mlflow=False)

    intervals = result.model.predict_interval(df.head(10))
    assert intervals is not None
    assert intervals.shape == (10, 2)
    assert (intervals[:, 1] >= intervals[:, 0]).all()  # ci_high >= ci_low


def test_bayesian_saved_model_roundtrip(tmp_path, valid_readmission_df):
    from mlops_core.models import load_model

    cfg = _small_cfg(tmp_path)
    df = pd.concat([valid_readmission_df] * 5, ignore_index=True)
    result = train_model(cfg, df, artifacts_dir=str(tmp_path), log_to_mlflow=False)

    reloaded = load_model(result.model_path)
    p1 = result.model.predict_proba(df.head(5))
    p2 = reloaded.predict_proba(df.head(5))
    assert (np.abs(p1 - p2) < 1e-9).all()
