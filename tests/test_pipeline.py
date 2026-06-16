"""La orquestación end-to-end encadena las etapas sin error (sin Spark, vía Parquet)."""

from __future__ import annotations

import sys
from pathlib import Path

from mlops_core.config import load_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_synthetic import generate  # noqa: E402
from run_pipeline import run_pipeline  # noqa: E402


def test_run_pipeline_end_to_end(tmp_path):
    cfg = load_config(ROOT / "configs" / "fraud.yaml")
    cfg.model.params["n_estimators"] = 60
    cfg.data.parquet_path = str(tmp_path / "fraud.parquet")

    # Escribe un Parquet pequeño para saltar la ingesta Spark (skip_ingest).
    generate(rows=4000, seed=9, fraud_rate=0.12).to_parquet(cfg.data.parquet_path)

    # No debe levantar excepción; ejercita validación + train + drift.
    run_pipeline(
        cfg,
        skip_ingest=True,
        log_to_mlflow=False,
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    assert (tmp_path / "artifacts" / "fraud" / "model.joblib").exists()
