"""Esquema Pandera del dominio banca/fraude.

Es el contrato de datos: si la data no lo cumple, fallamos antes del modelo. Valida
tipos, rangos (monto >= 0, coordenadas en rango), categorías permitidas y el label binario.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# Categorías estables del dataset Sparkov de fraude de tarjetas.
FRAUD_CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]

fraud_schema = DataFrameSchema(
    columns={
        "trans_date_trans_time": Column(pa.DateTime, nullable=False),
        "amt": Column(float, Check.ge(0), nullable=False),
        "category": Column(str, Check.isin(FRAUD_CATEGORIES), nullable=False),
        "gender": Column(str, Check.isin(["M", "F"]), nullable=False),
        "state": Column(str, nullable=False),
        "city_pop": Column(int, Check.ge(0), nullable=False),
        "lat": Column(float, Check.in_range(-90, 90), nullable=False),
        "long": Column(float, Check.in_range(-180, 180), nullable=False),
        "merch_lat": Column(float, Check.in_range(-90, 90), nullable=False),
        "merch_long": Column(float, Check.in_range(-180, 180), nullable=False),
        "is_fraud": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,  # se permiten columnas extra del dataset crudo
    coerce=True,
    name="fraud",
)
