"""Esquema Pandera del dominio educación — riesgo de deserción estudiantil.

Dataset UCI 'Predict Students Dropout and Academic Success' (id=697). El target
binario `dropout` vale 1 si el estudiante desertó, 0 si se graduó o continúa.

El drift de distribuciones detecta cambios de cohorte año a año — eso es exactamente
el argumento de 'detección de señales aplicada al modelo' que conecta con la
experiencia en farmacovigilancia del autor.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

dropout_schema = DataFrameSchema(
    columns={
        "Marital Status": Column(int, Check.isin([1, 2, 3, 4, 5, 6]), nullable=False),
        "Application mode": Column(int, Check.ge(1), nullable=False),
        "Application order": Column(int, Check.in_range(0, 9), nullable=False),
        "Course": Column(int, Check.ge(1), nullable=False),
        "Daytime/evening attendance": Column(int, Check.isin([0, 1]), nullable=False),
        "Previous qualification": Column(int, Check.ge(1), nullable=False),
        "Previous qualification (grade)": Column(float, Check.in_range(0, 200), nullable=False),
        "Admission grade": Column(float, Check.in_range(0, 200), nullable=False),
        "Age at enrollment": Column(int, Check.in_range(15, 70), nullable=False),
        "Gender": Column(int, Check.isin([0, 1]), nullable=False),
        "Scholarship holder": Column(int, Check.isin([0, 1]), nullable=False),
        "Debtor": Column(int, Check.isin([0, 1]), nullable=False),
        "Tuition fees up to date": Column(int, Check.isin([0, 1]), nullable=False),
        "Curricular units 1st sem (approved)": Column(int, Check.ge(0), nullable=False),
        "Curricular units 2nd sem (approved)": Column(int, Check.ge(0), nullable=False),
        "Curricular units 1st sem (grade)": Column(float, Check.in_range(0, 20), nullable=False),
        "Curricular units 2nd sem (grade)": Column(float, Check.in_range(0, 20), nullable=False),
        "Unemployment rate": Column(float, nullable=False),
        "Inflation rate": Column(float, nullable=False),
        "GDP": Column(float, nullable=False),
        "dropout": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,
    coerce=True,
    name="dropout",
)
