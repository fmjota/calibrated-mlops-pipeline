"""Carga y validación del config de dominio.

Cada dominio (fraude, reingreso, deserción) se describe con un YAML. El núcleo del
pipeline es agnóstico al dominio: lee este config para saber qué columnas usar, cómo
partir en el tiempo, qué umbrales aplicar, etc. Validar el config con Pydantic es la
primera línea de "fallar temprano": si el YAML está mal, fallamos antes de tocar datos.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    raw_path: str
    parquet_path: str


class Columns(BaseModel):
    datetime: str | None = None
    amount: str | None = None
    categorical: list[str] = Field(default_factory=list)
    numeric: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    type: str = "lightgbm"
    params: dict = Field(default_factory=dict)
    time_split_column: str | None = None
    valid_fraction: float = 0.2


class CalibrationConfig(BaseModel):
    # "auto" elige entre isotónica y Platt por mejor Brier en validación.
    method: str = "auto"


class EvaluationConfig(BaseModel):
    # Precisión objetivo desde la que se deriva el umbral de bloqueo/intervención.
    target_precision: float = 0.9


class DriftConfig(BaseModel):
    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.05


class DomainConfig(BaseModel):
    domain: str
    description: str = ""
    data: DataPaths
    target: str
    positive_label: int = 1
    columns: Columns = Field(default_factory=Columns)
    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)

    @property
    def feature_columns(self) -> list[str]:
        """Columnas de entrada al modelo (numéricas + categóricas, sin duplicados)."""
        seen: dict[str, None] = {}
        for col in [*self.columns.numeric, *self.columns.categorical]:
            seen.setdefault(col, None)
        return list(seen)


def load_config(path: str | Path) -> DomainConfig:
    """Lee un YAML de dominio y lo valida contra el esquema Pydantic.

    Args:
        path: ruta al archivo YAML del dominio (ej. `configs/fraud.yaml`).

    Returns:
        DomainConfig: config validado y tipado.

    Raises:
        pydantic.ValidationError: si el YAML no cumple el esquema.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DomainConfig.model_validate(raw)
