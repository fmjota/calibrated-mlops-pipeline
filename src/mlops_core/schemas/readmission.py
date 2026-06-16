"""Esquema Pandera del dominio salud — reingreso hospitalario (<30 días).

El dataset es UCI 'Diabetes 130-US Hospitals'. El target binario `readmitted_30d` vale 1
si el paciente fue readmitido en menos de 30 días (umbral clínico relevante: reingresos
tempranos son los más costosos y prevenibles).

Solo se validan las columnas usadas por el modelo; columnas extra del CSV (diag_*, ids)
se permiten con `strict=False`.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

MEDICATION_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "insulin",
]
MED_VALUES = ["No", "Down", "Steady", "Up"]

AGE_BUCKETS = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
RACE_VALUES = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
GENDER_VALUES = ["Female", "Male", "Unknown/Invalid"]

readmission_schema = DataFrameSchema(
    columns={
        "race": Column(str, Check.isin(RACE_VALUES), nullable=True),
        "gender": Column(str, Check.isin(GENDER_VALUES), nullable=False),
        "age": Column(str, Check.isin(AGE_BUCKETS), nullable=False),
        "time_in_hospital": Column(int, Check.in_range(1, 14), nullable=False),
        "num_lab_procedures": Column(int, Check.ge(0), nullable=False),
        "num_procedures": Column(int, Check.in_range(0, 6), nullable=False),
        "num_medications": Column(int, Check.ge(0), nullable=False),
        "number_outpatient": Column(int, Check.ge(0), nullable=False),
        "number_emergency": Column(int, Check.ge(0), nullable=False),
        "number_inpatient": Column(int, Check.ge(0), nullable=False),
        "number_diagnoses": Column(int, Check.in_range(1, 16), nullable=False),
        "change": Column(str, Check.isin(["No", "Ch"]), nullable=False),
        "diabetesMed": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        **{col: Column(str, Check.isin(MED_VALUES), nullable=False) for col in MEDICATION_COLS},
        "readmitted_30d": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,
    coerce=True,
    name="readmission",
)
