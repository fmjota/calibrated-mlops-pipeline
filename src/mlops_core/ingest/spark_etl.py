"""ETL de ingesta con PySpark: CSV crudo -> Parquet tipado.

Spark se justifica por el volumen del dataset (~1.85M filas en el caso real). La ingesta
tipa columnas y parsea la fecha; el feature engineering pesado vive en la capa `features`.
La validación Pandera corre después, sobre el Parquet ya tipado.
"""

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from mlops_core.config import DomainConfig


def ingest_csv_to_parquet(
    cfg: DomainConfig,
    spark: SparkSession,
    *,
    mode: str = "overwrite",
) -> str:
    """Lee el CSV crudo, tipa columnas y parsea la fecha, y escribe Parquet.

    Args:
        cfg: config del dominio; usa `data.raw_path`, `data.parquet_path` y `columns.datetime`.
        spark: SparkSession activa (ver `mlops_core.spark.get_spark`).
        mode: modo de escritura de Spark ("overwrite" por defecto).

    Returns:
        str: ruta del directorio Parquet generado (igual a `cfg.data.parquet_path`).

    Efectos secundarios:
        Escribe un dataset Parquet en disco en `cfg.data.parquet_path`.
    """
    df = spark.read.csv(cfg.data.raw_path, header=True, inferSchema=True)

    dt_col = cfg.columns.datetime
    if dt_col and dt_col in df.columns:
        df = df.withColumn(dt_col, F.to_timestamp(F.col(dt_col)))

    df.write.mode(mode).parquet(cfg.data.parquet_path)
    return cfg.data.parquet_path


def read_parquet_pandas(cfg: DomainConfig):
    """Lee el Parquet generado como pandas (vía pyarrow), sin necesitar Spark."""
    import pandas as pd

    return pd.read_parquet(cfg.data.parquet_path)
