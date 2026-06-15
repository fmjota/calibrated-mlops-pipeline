"""Ingesta/ETL distribuido con PySpark."""

from mlops_core.ingest.spark_etl import ingest_csv_to_parquet, read_parquet_pandas

__all__ = ["ingest_csv_to_parquet", "read_parquet_pandas"]
