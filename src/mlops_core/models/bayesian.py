"""Regresión logística bayesiana con PyMC.

La clave de esta variante no es el algoritmo en sí (LightGBM ya da buenas probabilidades),
sino la **cuantificación explícita de incertidumbre**: una estimación con su intervalo
creíble es más defendible ante un comité clínico que un número pelado. Dice explícitamente
"no sé" donde los datos son escasos.

Diseño:
- Priors débiles N(0,1) sobre coeficientes estandarizados (regularización suave).
- Posterior vía NUTS (sampler HMC de PyMC). Para escalar, se usa un subsample
  estratificado de `sample_size` filas antes del sampling.
- `predict_proba()`: media de la posterior predictiva (calibrada por construcción).
- `predict_interval(level)`: cuantiles de la posterior predictiva por fila
  → intervalo creíble del nivel pedido.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pymc as pm


def _build_design_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    means_stds: dict | None = None,
    cat_dummies: list[str] | None = None,
) -> tuple[np.ndarray, dict, list[str]]:
    """Construye la matriz de diseño numérica para PyMC.

    Args:
        df: datos de entrada.
        numeric_cols: columnas numéricas (se estandarizan).
        categorical_cols: columnas categóricas (one-hot, baja cardinalidad).
        means_stds: dict {col: (mean, std)} del entrenamiento; None → calculamos aquí.
        cat_dummies: lista de columnas dummy del entrenamiento; None → creamos aquí.

    Returns:
        (X_np, means_stds, cat_dummies): matriz float64 [n, p], stats de normalización
        y lista de columnas dummy para reaplicar en inferencia.
    """
    parts = []

    # Numéricas estandarizadas
    ms: dict[str, tuple[float, float]] = means_stds or {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        vals = df[col].fillna(0).astype(float)
        if means_stds is None:
            mu, sigma = float(vals.mean()), float(vals.std()) or 1.0
            ms[col] = (mu, sigma)
        else:
            mu, sigma = ms[col]
        parts.append(((vals - mu) / sigma).values.reshape(-1, 1))

    # Categóricas one-hot
    cat_df = df[categorical_cols].fillna("Unknown").astype(str)
    dummies = pd.get_dummies(cat_df, drop_first=True)
    if cat_dummies is not None:
        dummies = dummies.reindex(columns=cat_dummies, fill_value=0)
    else:
        cat_dummies = list(dummies.columns)
    parts.append(dummies.values.astype(float))

    X = np.hstack(parts) if parts else np.zeros((len(df), 0))
    return X.astype(np.float64), ms, cat_dummies


@dataclass
class BayesianLogisticModel:
    """Regresión logística bayesiana: entrena con NUTS, predice media + intervalos.

    Attrs:
        numeric_cols: columnas numéricas del dominio (se estandarizan en entrenamiento).
        categorical_cols: columnas categóricas (one-hot).
        ci_level: nivel del intervalo creíble a reportar (ej. 0.90 → IC 90%).
    """

    numeric_cols: list[str]
    categorical_cols: list[str]
    ci_level: float = 0.90

    # Fijados en fit():
    _trace: object = field(default=None, repr=False)
    _means_stds: dict = field(default_factory=dict, repr=False)
    _cat_dummies: list[str] = field(default_factory=list, repr=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        draws: int = 500,
        tune: int = 200,
        chains: int = 2,
        target_accept: float = 0.9,
        sample_size: int = 10_000,
        random_seed: int = 42,
    ) -> BayesianLogisticModel:
        """Ajusta la posterior vía NUTS con un subsample estratificado.

        Args:
            X: features (pandas DataFrame).
            y: target binario (pandas Series de int).
            draws: muestras NUTS por cadena.
            tune: pasos de warm-up.
            chains: cadenas paralelas.
            target_accept: tasa de aceptación objetivo del NUTS.
            sample_size: máximo de filas para NUTS (subsample estratificado).
            random_seed: semilla de reproducibilidad.

        Returns:
            self (para encadenar).

        Efectos secundarios:
            Almacena la traza de PyMC en `self._trace`.
        """
        rng = np.random.default_rng(random_seed)

        # Subsample estratificado para escalar NUTS a datasets medianos
        y_arr = np.asarray(y)
        n = len(y_arr)
        if n > sample_size:
            pos_idx = np.flatnonzero(y_arr == 1)
            neg_idx = np.flatnonzero(y_arr == 0)
            n_pos = min(len(pos_idx), int(sample_size * y_arr.mean()) + 1)
            n_neg = sample_size - n_pos
            idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=n_pos, replace=False),
                    rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
                ]
            )
            X_sub = X.iloc[idx].reset_index(drop=True)
            y_sub = y.iloc[idx].reset_index(drop=True)
        else:
            X_sub, y_sub = X, y

        X_np, self._means_stds, self._cat_dummies = _build_design_matrix(
            X_sub, self.numeric_cols, self.categorical_cols
        )
        y_np = np.asarray(y_sub, dtype=np.int32)
        n_features = X_np.shape[1]

        with pm.Model():
            alpha = pm.Normal("alpha", mu=0, sigma=2)
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)
            logit_p = alpha + pm.math.dot(X_np, beta)
            pm.Bernoulli("obs", logit_p=logit_p, observed=y_np)
            self._trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=False,
            )
        return self

    def _posterior_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Muestras de la posterior predictiva por fila. Shape: [n_samples, n_rows]."""
        X_np, _, _ = _build_design_matrix(
            X,
            self.numeric_cols,
            self.categorical_cols,
            means_stds=self._means_stds,
            cat_dummies=self._cat_dummies,
        )
        alpha = self._trace.posterior["alpha"].values.reshape(-1)  # [S]
        beta = self._trace.posterior["beta"].values.reshape(-1, X_np.shape[1])  # [S, p]
        logits = alpha[:, None] + beta @ X_np.T  # [S, n]
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Media de la posterior predictiva por fila (probabilidad calibrada).

        Args:
            X: features como pandas DataFrame.

        Returns:
            np.ndarray de shape [n_rows] con la probabilidad de la clase positiva.
        """
        return self._posterior_samples(X).mean(axis=0)

    def predict_interval(self, X: pd.DataFrame, level: float | None = None) -> np.ndarray:
        """Intervalo creíble de la posterior predictiva por fila.

        Args:
            X: features como pandas DataFrame.
            level: nivel del intervalo (ej. 0.90 → IC 90%). Si None, usa `self.ci_level`.

        Returns:
            np.ndarray de shape [n_rows, 2] con (ci_low, ci_high) por fila.
        """
        level = level or self.ci_level
        tail = (1.0 - level) / 2.0
        samples = self._posterior_samples(X)  # [S, n]
        low = np.quantile(samples, tail, axis=0)
        high = np.quantile(samples, 1 - tail, axis=0)
        return np.stack([low, high], axis=1)
