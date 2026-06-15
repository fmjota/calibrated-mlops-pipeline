"""Ingesta PySpark end-to-end: genera CSV -> ETL a Parquet -> validación Pandera."""

from __future__ import annotations

from pathlib import Path

from mlops_core.config import load_config
from mlops_core.ingest import ingest_csv_to_parquet, read_parquet_pandas
from mlops_core.validate import validate_dataframe

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_ingest_then_validate(spark, tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_synthetic import generate

    csv_path = tmp_path / "raw.csv"
    generate(rows=2000, seed=7).to_csv(csv_path, index=False)

    cfg = load_config(CONFIGS / "fraud.yaml")
    cfg.data.raw_path = str(csv_path)
    cfg.data.parquet_path = str(tmp_path / "fraud.parquet")

    out_path = ingest_csv_to_parquet(cfg, spark)
    assert Path(out_path).exists()

    pdf = read_parquet_pandas(cfg)
    assert len(pdf) == 2000

    validated = validate_dataframe(pdf, cfg.domain)
    assert "is_fraud" in validated.columns
    assert validated["is_fraud"].isin([0, 1]).all()
