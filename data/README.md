# Datos

Los datasets **no se versionan** (ver `.gitignore`). Se descargan con:

```bash
bash scripts/download_data.sh
```

## Dominio banca/fraude (dataset inicial)

*Credit Card Transactions Fraud Detection* (simulado con Sparkov). Esquema tabular rico
(monto, comercio, categoría, geo, tiempo) y desbalance fuerte (~0.5% de fraude), ideal
para demostrar validación, calibración y manejo de clases desbalanceadas.

Estructura esperada tras la descarga:

```
data/
├── raw/         # CSV crudos descargados
└── processed/   # Parquet generados por el ETL PySpark
```
