"""Descarga el dataset UCI 'Predict Students Dropout and Academic Success' (id=697).

El target se binariza: `dropout` = 1 si el estudiante desertó, 0 si continúa inscrito
o se graduó. Esto refleja el caso de uso: intervenir antes de que ocurra la deserción.

Args (CLI):
    --out: ruta de salida del CSV (por defecto data/raw/dropout_train.csv).

Efectos secundarios:
    Escribe un CSV en disco. Requiere conexión a internet la primera vez.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_and_prepare(out_path: str = "data/raw/dropout_train.csv") -> pd.DataFrame:
    """Descarga, binariza el target y guarda el CSV. Devuelve el DataFrame."""
    print("Descargando UCI Predict Students Dropout and Academic Success (id=697)…")
    dataset = fetch_ucirepo(id=697)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    # El target original tiene 3 valores: 'Dropout', 'Enrolled', 'Graduate'.
    # Se binariza: 1 = Dropout (el evento a predecir), 0 = cualquier otro.
    target_col = y.columns[0]
    df["dropout"] = (df[target_col] == "Dropout").astype(int)
    df = df.drop(columns=[target_col])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_pos = df["dropout"].sum()
    print(f"Guardado: {out} — {len(df):,} filas, {n_pos:,} desertores ({n_pos / len(df):.2%})")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga dataset de deserción estudiantil.")
    parser.add_argument("--out", default="data/raw/dropout_train.csv")
    args = parser.parse_args()
    download_and_prepare(args.out)


if __name__ == "__main__":
    main()
