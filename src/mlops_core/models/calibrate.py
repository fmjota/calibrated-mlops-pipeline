"""Calibración de probabilidades: isotónica o Platt (sigmoide), elegida por Brier.

Un clasificador puede rankear bien (buen ROC) pero entregar probabilidades sesgadas. Si
la decisión opera sobre un umbral (bloquear una transacción, intervenir a un paciente),
la probabilidad tiene que estar **calibrada**: que "0.8" signifique ~80% de las veces.
Calibramos sobre un set aparte y elegimos el método que minimiza el Brier.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from mlops_core.evaluate.metrics import brier

_METHODS = ("isotonic", "sigmoid")


class Calibrator:
    """Mapea probabilidades crudas a calibradas. Métodos: isotonic, sigmoid, identity."""

    def __init__(self, method: str = "isotonic"):
        if method not in (*_METHODS, "identity"):
            raise ValueError(f"Método de calibración inválido: {method!r}")
        self.method = method
        self._model: object | None = None

    def fit(self, p, y) -> Calibrator:
        """Ajusta el calibrador. Inputs: probas crudas `p` y labels `y` (0/1). Devuelve self."""
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        elif self.method == "sigmoid":
            self._model = LogisticRegression().fit(p.reshape(-1, 1), y)
        return self

    def transform(self, p) -> np.ndarray:
        """Mapea probas crudas `p` a calibradas. Devuelve un array float del mismo largo."""
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return np.asarray(self._model.predict(p), dtype=float)
        if self.method == "sigmoid":
            return self._model.predict_proba(p.reshape(-1, 1))[:, 1]
        return p


def fit_best_calibrator(
    p_cal, y_cal, p_eval, y_eval, method: str = "auto"
) -> tuple[Calibrator, str]:
    """Ajusta calibradores en (p_cal, y_cal) y elige el de menor Brier en el set de eval.

    `method="auto"` compara isotónica vs sigmoide; otro valor fuerza el método.
    """
    candidates = list(_METHODS) if method == "auto" else [method]
    fitted = {m: Calibrator(m).fit(p_cal, y_cal) for m in candidates}
    best = min(candidates, key=lambda m: brier(y_eval, fitted[m].transform(p_eval)))
    return fitted[best], best
