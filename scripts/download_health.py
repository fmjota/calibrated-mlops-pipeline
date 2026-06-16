"""Descarga el dataset UCI 'Diabetes 130-US Hospitals' (id=296) y lo prepara.

El target binario es `readmitted_30d`: 1 si el paciente fue readmitido en <30 días,
0 en cualquier otro caso (>30 días o sin readmisión). Este criterio refleja el umbral
clínico relevante: reingresos tempranos son los más costosos y prevenibles.

Args (CLI):
    --out: ruta de salida del CSV (por defecto data/raw/readmission_train.csv).

Efectos secundarios:
    Escribe un CSV en disco. Requiere conexión a internet la primera vez.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_and_prepare(out_path: str = "data/raw/readmission_train.csv") -> pd.DataFrame:
    """Descarga, limpia y binariza el target. Devuelve el DataFrame listo."""
    print("Descargando UCI Diabetes 130-US Hospitals (id=296)…")
    dataset = fetch_ucirepo(id=296)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    # Target binario: <30 días = readmisión temprana (evento de interés).
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Limpieza básica: reemplazar '?' por NaN y quitar columnas de alta cardinalidad.
    df = df.replace("?", np.nan)
    drop_cols = [c for c in ["payer_code", "medical_specialty", "weight"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Convertir columnas numéricas
    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["readmitted_30d"])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_pos = df["readmitted_30d"].sum()
    print(
        f"Guardado: {out} — {len(df):,} filas, {n_pos:,} readmisiones <30d ({n_pos / len(df):.2%})"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga dataset de reingreso hospitalario.")
    parser.add_argument("--out", default="data/raw/readmission_train.csv")
    args = parser.parse_args()
    download_and_prepare(args.out)


if __name__ == "__main__":
    main()
