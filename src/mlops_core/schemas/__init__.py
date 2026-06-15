"""Registro de esquemas Pandera por dominio."""

from __future__ import annotations

from pandera.pandas import DataFrameSchema

from mlops_core.schemas.fraud import fraud_schema

_REGISTRY: dict[str, DataFrameSchema] = {
    "fraud": fraud_schema,
}


def get_schema(domain: str) -> DataFrameSchema:
    """Devuelve el esquema del dominio; falla claro si no está registrado."""
    try:
        return _REGISTRY[domain]
    except KeyError:
        disponibles = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"No hay esquema registrado para el dominio '{domain}'. Disponibles: {disponibles}."
        ) from None


__all__ = ["fraud_schema", "get_schema"]
