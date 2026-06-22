# Datos

Los datasets no se versionan (ver `.gitignore`). Se descargan con:

```bash
bash scripts/download_data.sh
```

## Dominio banca/fraude (dataset inicial)

*Credit Card Transactions Fraud Detection* (simulado con Sparkov). Esquema tabular
(monto, comercio, categoría, geo, tiempo) con desbalance fuerte (~0.5% de fraude), útil
para demostrar validación, calibración y manejo de clases desbalanceadas.

Estructura esperada tras la descarga:

```
data/
├── raw/         # CSV crudos descargados
└── processed/   # Parquet generados por el ETL PySpark
```
