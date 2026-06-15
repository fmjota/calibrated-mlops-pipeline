from pathlib import Path

import pytest
from pydantic import ValidationError

from mlops_core.config import DomainConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_load_fraud_config():
    cfg = load_config(CONFIGS / "fraud.yaml")
    assert isinstance(cfg, DomainConfig)
    assert cfg.domain == "fraud"
    assert cfg.target == "is_fraud"
    assert cfg.positive_label == 1
    assert cfg.columns.amount == "amt"
    assert "category" in cfg.columns.categorical
    assert cfg.model.type == "lightgbm"
    assert cfg.model.params["is_unbalance"] is True
    assert cfg.drift.psi_threshold == pytest.approx(0.2)


def test_feature_columns_dedup():
    cfg = load_config(CONFIGS / "fraud.yaml")
    # "amt" aparece en numeric; no debe duplicarse al combinar listas.
    assert cfg.feature_columns.count("amt") == 1
    assert set(cfg.columns.categorical).issubset(set(cfg.feature_columns))


def test_invalid_config_fails_fast():
    with pytest.raises(ValidationError):
        DomainConfig.model_validate({"domain": "x"})  # faltan data/target
