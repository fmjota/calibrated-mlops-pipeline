"""Contratos de la API: request y response.

El request valida en el **borde** del sistema (mismo espíritu que Pandera en el ETL):
una transacción mal formada se rechaza con 422 antes de tocar el modelo.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from mlops_core.schemas.fraud import FRAUD_CATEGORIES


class Transaction(BaseModel):
    """Una transacción de tarjeta con los campos que el modelo necesita."""

    trans_date_trans_time: datetime
    amt: float = Field(ge=0)
    category: str
    gender: str
    state: str
    city_pop: int = Field(ge=0)
    lat: float = Field(ge=-90, le=90)
    long: float = Field(ge=-180, le=180)
    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)

    @field_validator("category")
    @classmethod
    def _valid_category(cls, v: str) -> str:
        if v not in FRAUD_CATEGORIES:
            raise ValueError(f"category inválida: {v!r}")
        return v

    @field_validator("gender")
    @classmethod
    def _valid_gender(cls, v: str) -> str:
        if v not in ("M", "F"):
            raise ValueError("gender debe ser 'M' o 'F'")
        return v


class PredictionResponse(BaseModel):
    probability: float = Field(description="Probabilidad calibrada de la clase positiva.")
    decision: int = Field(description="1 si probability >= umbral, 0 si no.")
    threshold: float = Field(description="Umbral de decisión del modelo.")
    domain: str
    ci_low: float | None = Field(
        default=None, description="Extremo inferior del intervalo creíble (solo modelo bayesiano)."
    )
    ci_high: float | None = Field(
        default=None, description="Extremo superior del intervalo creíble (solo modelo bayesiano)."
    )
