"""Métricas honestas: PR-AUC, Brier, umbral por precisión objetivo."""

from __future__ import annotations

import numpy as np

from mlops_core.evaluate.metrics import (
    brier,
    metrics_at_threshold,
    pr_auc,
    reliability_curve,
    threshold_for_precision,
)


def test_pr_auc_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y, p) == 1.0
    assert brier(y, p) < 0.05


def test_threshold_reaches_target_precision():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    # Score con señal: positivos más altos que negativos en promedio.
    p = np.concatenate([rng.beta(2, 8, 900), rng.beta(8, 2, 100)])
    thr = threshold_for_precision(y, p, target_precision=0.8)
    m = metrics_at_threshold(y, p, thr)
    assert 0.0 <= thr <= 1.0
    assert m["precision"] >= 0.8 - 1e-9


def test_threshold_unreachable_returns_one():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.4, 0.4, 0.4, 0.4])
    assert threshold_for_precision(y, p, target_precision=0.99) == 1.0


def test_reliability_curve_shapes():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(size=500)
    pred, true = reliability_curve(y, p, n_bins=5)
    assert len(pred) == len(true)
