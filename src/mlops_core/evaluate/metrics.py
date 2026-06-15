"""Métricas para clasificación con desbalance fuerte.

Para fraude (clases muy desbalanceadas) accuracy engaña: predecir "todo legítimo" da
>99%. Usamos **PR-AUC** (precisión-recall) y **Brier** (calidad de la probabilidad). El
umbral de decisión se deriva de una **precisión objetivo**: el umbral significa algo.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def pr_auc(y_true, p) -> float:
    """Área bajo la curva precisión-recall (average precision)."""
    return float(average_precision_score(y_true, p))


def roc_auc(y_true, p) -> float:
    return float(roc_auc_score(y_true, p))


def brier(y_true, p) -> float:
    """Brier score: error cuadrático medio de la probabilidad. Menor es mejor."""
    return float(brier_score_loss(y_true, p))


def threshold_for_precision(y_true, p, target_precision: float) -> float:
    """Menor umbral que alcanza la precisión objetivo, maximizando recall.

    Si ningún umbral alcanza esa precisión, devuelve 1.0 (no marcar nada).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, p)
    # precision/recall tienen un punto más que thresholds; alineamos con [:-1].
    prec, rec, thr = precision[:-1], recall[:-1], thresholds
    ok = np.flatnonzero(prec >= target_precision)
    if ok.size == 0:
        return 1.0
    best = ok[np.argmax(rec[ok])]
    return float(thr[best])


def reliability_curve(y_true, p, n_bins: int = 10) -> tuple[list[float], list[float]]:
    """Puntos (prob_predicha, frac_observada) para el diagrama de fiabilidad."""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
    return prob_pred.tolist(), prob_true.tolist()


def metrics_at_threshold(y_true, p, threshold: float) -> dict[str, float]:
    """Precisión, recall, F1 y nº de casos marcados a un umbral dado."""
    y_true = np.asarray(y_true)
    y_hat = (np.asarray(p) >= threshold).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_flagged": float(int(y_hat.sum())),
    }
