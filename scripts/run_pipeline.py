"""Orquesta el pipeline de MLOps de punta a punta para un dominio.

Encadena las etapas del núcleo dirigido por `config`:
ingesta (PySpark) → validación (Pandera) → entrenamiento + calibración (LightGBM/MLflow)
→ detección de drift (PSI/KS). Imprime un resumen legible de cada etapa.

Uso:
    uv run python scripts/run_pipeline.py --config configs/fraud.yaml

Args (CLI):
    --config: ruta al YAML del dominio (obligatorio).
    --skip-ingest: reutiliza el Parquet existente en vez de releer el CSV con Spark.
    --no-mlflow: no registra la corrida en MLflow (útil en demos rápidas).

Efectos secundarios:
    Escribe Parquet (ingesta), el modelo serializado en `artifacts/<domain>/` y,
    salvo `--no-mlflow`, una corrida en `mlruns/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlops_core.config import DomainConfig, load_config
from mlops_core.drift import detect_drift
from mlops_core.ingest import ingest_csv_to_parquet, read_parquet_pandas
from mlops_core.models import train_model
from mlops_core.validate import validate_dataframe


def _section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def _run_ingest(cfg: DomainConfig) -> None:
    from mlops_core.spark import get_spark

    spark = get_spark(app_name=f"run-pipeline-{cfg.domain}")
    try:
        ingest_csv_to_parquet(cfg, spark)
    finally:
        spark.stop()


def run_pipeline(
    cfg: DomainConfig,
    *,
    skip_ingest: bool,
    log_to_mlflow: bool,
    artifacts_dir: str = "artifacts",
) -> None:
    _section(f"PIPELINE — dominio '{cfg.domain}'")
    print(cfg.description or "(sin descripción)")

    # 1. Ingesta CSV -> Parquet (PySpark)
    _section("1/4 · Ingesta (PySpark)")
    if skip_ingest and Path(cfg.data.parquet_path).exists():
        print(f"Reutilizando Parquet existente: {cfg.data.parquet_path}")
    else:
        if not Path(cfg.data.raw_path).exists():
            raise FileNotFoundError(
                f"No existe {cfg.data.raw_path}. Corre antes: bash scripts/download_data.sh"
            )
        _run_ingest(cfg)
        print(f"CSV → Parquet: {cfg.data.parquet_path}")

    pdf = read_parquet_pandas(cfg)
    print(f"Filas: {len(pdf):,} | columnas: {len(pdf.columns)}")

    # 2. Validación (Pandera) — falla temprano y trazable
    _section("2/4 · Validación (Pandera)")
    pdf = validate_dataframe(pdf, cfg.domain)
    print(f"Contrato OK · prevalencia positiva: {pdf[cfg.target].mean():.3%}")

    # 3. Entrenamiento + calibración + métricas (LightGBM / MLflow)
    _section("3/4 · Entrenamiento, calibración y métricas")
    result = train_model(cfg, pdf, artifacts_dir=artifacts_dir, log_to_mlflow=log_to_mlflow)
    m = result.metrics
    print(f"PR-AUC  crudo→cal : {m['pr_auc_raw']:.3f} → {m['pr_auc_cal']:.3f}")
    print(f"Brier   crudo→cal : {m['brier_raw']:.4f} → {m['brier_cal']:.4f}  (menor es mejor)")
    print(f"ROC-AUC           : {m['roc_auc']:.3f}")
    print(f"Calibración elegida: {result.calibration_method}")
    print(
        f"Umbral de bloqueo : {result.threshold:.4f} "
        f"(precisión {m['precision_at_threshold']:.2f}, recall {m['recall_at_threshold']:.2f})"
    )
    print(f"Modelo guardado   : {result.model_path}")

    # 4. Drift: referencia (pasado) vs producción (slice reciente por tiempo)
    _section("4/4 · Detección de drift (PSI / KS)")
    dtc = cfg.model.time_split_column or cfg.columns.datetime
    pdf_sorted = pdf.sort_values(dtc) if dtc and dtc in pdf.columns else pdf
    cut = int(len(pdf_sorted) * 0.7)
    reference, current = pdf_sorted.iloc[:cut], pdf_sorted.iloc[cut:]
    report = detect_drift(reference, current, cfg)
    print(f"Referencia: {len(reference):,} filas | Producción: {len(current):,} filas")
    print(report.summary())
    print(report.to_frame().round(4).to_string(index=False))

    _section("FIN — pipeline completo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de MLOps end-to-end por dominio.")
    parser.add_argument("--config", required=True, help="Ruta al YAML del dominio.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reusar Parquet existente.")
    parser.add_argument("--no-mlflow", action="store_true", help="No registrar en MLflow.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, skip_ingest=args.skip_ingest, log_to_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
