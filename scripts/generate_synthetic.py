"""Genera un dataset de fraude sintético con el mismo esquema que el dataset Sparkov.

Sirve para que el pipeline corra en cualquier lado (tests, CI, demo) sin depender de
credenciales de Kaggle. El dataset real se descarga con `scripts/download_data.sh`; este
generador produce datos con la misma forma (columnas, tipos, desbalance ~0.6%).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_core.schemas.fraud import FRAUD_CATEGORIES

STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "WA", "CO", "AZ"]


def generate(rows: int = 200_000, seed: int = 42, fraud_rate: float = 0.006) -> pd.DataFrame:
    """Devuelve un DataFrame con el esquema del dominio fraude."""
    rng = np.random.default_rng(seed)

    start = np.datetime64("2019-01-01T00:00:00")
    offsets = np.sort(rng.integers(0, 86_400 * 540, size=rows))  # ~18 meses, ordenado
    ts = start + offsets.astype("timedelta64[s]")

    category = rng.choice(FRAUD_CATEGORIES, size=rows)
    gender = rng.choice(["M", "F"], size=rows)
    state = rng.choice(STATES, size=rows)
    city_pop = rng.integers(500, 3_000_000, size=rows)
    lat = rng.uniform(25.0, 49.0, size=rows)
    lon = rng.uniform(-124.0, -67.0, size=rows)
    merch_lat = lat + rng.normal(0, 0.3, size=rows)
    merch_long = lon + rng.normal(0, 0.3, size=rows)

    amt = np.round(rng.lognormal(mean=3.2, sigma=1.1, size=rows), 2)

    # Probabilidad de fraude crece con el monto y en categorías "net".
    net = np.isin(category, ["misc_net", "shopping_net", "grocery_net"])
    logit = -7.0 + 0.0009 * amt + 0.8 * net + rng.normal(0, 0.5, size=rows)
    prob = 1.0 / (1.0 + np.exp(-logit))
    is_fraud = (rng.uniform(size=rows) < prob).astype(int)

    # Reescala para acercarnos a la tasa objetivo manteniendo la señal.
    target_n = int(rows * fraud_rate)
    if is_fraud.sum() > target_n:
        fraud_idx = np.flatnonzero(is_fraud)
        drop = rng.choice(fraud_idx, size=is_fraud.sum() - target_n, replace=False)
        is_fraud[drop] = 0

    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.to_datetime(ts),
            "cc_num": rng.integers(10**15, 10**16, size=rows),
            "merchant": [f"merch_{i}" for i in rng.integers(0, 800, size=rows)],
            "category": category,
            "amt": amt,
            "gender": gender,
            "state": state,
            "city_pop": city_pop,
            "lat": lat,
            "long": lon,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "unix_time": (ts - np.datetime64("1970-01-01T00:00:00")) // np.timedelta64(1, "s"),
            "is_fraud": is_fraud,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera dataset de fraude sintético.")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/fraud_train.csv")
    args = parser.parse_args()

    df = generate(rows=args.rows, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generadas {len(df):,} filas ({df['is_fraud'].mean():.3%} fraude) -> {out}")


if __name__ == "__main__":
    main()
