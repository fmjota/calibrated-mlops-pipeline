"""La detección de drift no marca datos estables y sí marca un corrimiento real."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mlops_core.config import load_config
from mlops_core.drift import detect_drift, psi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_synthetic import generate  # noqa: E402


def _cfg():
    return load_config(ROOT / "configs" / "fraud.yaml")


def test_psi_zero_on_identical():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    shifted = rng.normal(2, 1, 5000)
    assert psi(base, shifted) > 0.2


def test_no_drift_on_same_distribution():
    cfg = _cfg()
    ref = generate(rows=8000, seed=1)
    report = detect_drift(ref, ref.copy(), cfg)
    assert not report.drifted
    assert "Sin drift" in report.summary()


def test_drift_detected_on_shifted_amount():
    cfg = _cfg()
    ref = generate(rows=8000, seed=1)
    current = ref.copy()
    current["amt"] = current["amt"] * 3 + 50  # inflación/corrimiento del monto
    report = detect_drift(ref, current, cfg)
    assert report.drifted
    assert "amt" in report.drifted_features
    amt = next(f for f in report.features if f.feature == "amt")
    assert amt.psi > cfg.drift.psi_threshold
