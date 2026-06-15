"""La calibración mejora la calidad de la probabilidad (Brier)."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_core.evaluate.metrics import brier
from mlops_core.models.calibrate import Calibrator, fit_best_calibrator


def _miscalibrated(seed=0, n=4000):
    """Genera probabilidades crudas sesgadas (sobreconfiadas) y sus etiquetas."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    # Score crudo sesgado: empuja hacia los extremos (mal calibrado pero buen ranking).
    p_raw = np.clip(true_p**2.2, 1e-4, 1 - 1e-4)
    return p_raw, y


def test_isotonic_improves_brier():
    p_raw, y = _miscalibrated()
    n = len(y) // 2
    cal = Calibrator("isotonic").fit(p_raw[:n], y[:n])
    p_cal = cal.transform(p_raw[n:])
    assert brier(y[n:], p_cal) < brier(y[n:], p_raw[n:])


def test_fit_best_calibrator_picks_valid_method():
    p_raw, y = _miscalibrated(seed=3)
    n = len(y) // 2
    cal, method = fit_best_calibrator(p_raw[:n], y[:n], p_raw[n:], y[n:], method="auto")
    assert method in ("isotonic", "sigmoid")
    assert brier(y[n:], cal.transform(p_raw[n:])) <= brier(y[n:], p_raw[n:]) + 1e-6


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        Calibrator("inexistente")
