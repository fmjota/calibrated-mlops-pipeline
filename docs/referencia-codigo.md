# Referencia de código

Cada archivo del repo: **propósito** (una línea), **inputs** que consume, **outputs** que
produce y **depende de** qué otros módulos. Mantener sincronizado con los docstrings.

## `src/mlops_core/` — núcleo agnóstico al dominio

| Archivo | Propósito | Inputs | Outputs | Depende de |
|---|---|---|---|---|
| `config.py` | Cargar y validar el config de dominio (Pydantic) | ruta a YAML | `DomainConfig` | pydantic, yaml |
| `spark.py` | Crear SparkSession con un JDK 17/21 compatible | app_name | `SparkSession`; **efecto:** fija `JAVA_HOME` | pyspark |
| `schemas/fraud.py` | Esquema Pandera del dominio fraude + categorías válidas | — | `fraud_schema`, `FRAUD_CATEGORIES` | pandera |
| `schemas/__init__.py` | Registro de esquemas por dominio | `domain: str` | `DataFrameSchema` (`get_schema`) | schemas/fraud |
| `validate/runner.py` | Aplicar el contrato y fallar trazable | DataFrame, `domain` | DataFrame validado **o** `DataValidationError` | pandera, schemas |
| `ingest/spark_etl.py` | ETL CSV→Parquet tipado + leer Parquet a pandas | `DomainConfig`, `SparkSession` | **efecto:** escribe Parquet; ruta / DataFrame | pyspark, config |
| `features/build.py` | Features (fecha + haversine) consistentes train/inferencia | `DomainConfig`, DataFrame, `categories?` | `(X, y)`; dict de categorías | numpy, pandas, config |
| `evaluate/metrics.py` | Métricas honestas y selección de umbral | `y_true`, `p`, `target_precision` | floats / dict / curva | sklearn, numpy, scipy |
| `models/calibrate.py` | Calibrar probabilidades (isotónica/Platt) por Brier | probas+labels de calib y eval | `Calibrator`, método elegido | sklearn, evaluate |
| `models/calibrated_model.py` | Modelo serializable (base+calibrador+config) para serving | filas crudas | probas calibradas / decisión; **efecto:** joblib | joblib, features, calibrate |
| `models/train.py` | Entrenamiento end-to-end (split, LGBM, calib, métricas) | `DomainConfig`, DataFrame | `TrainResult`; **efecto:** guarda modelo + MLflow | lightgbm, todo el núcleo |
| `drift/detector.py` | PSI + KS por feature, referencia vs producción | reference, current, `DomainConfig` | `DriftReport` | numpy, scipy, config |
| `serve/schemas.py` | Contratos de la API (request/response) | JSON de transacción | `Transaction`, `PredictionResponse` | pydantic, schemas/fraud |
| `serve/api.py` | App FastAPI: `/health`, `/predict` | request HTTP; env `MODEL_PATH` | JSON con proba calibrada + decisión | fastapi, models, serve/schemas |

> Los `__init__.py` de cada subpaquete solo re-exportan la API pública del módulo.

## `scripts/`

| Archivo | Propósito | Inputs | Outputs |
|---|---|---|---|
| `generate_synthetic.py` | Generar datos de fraude con el esquema Sparkov | `--rows`, `--seed`, `--out` | CSV en `data/raw/`; función `generate()` |
| `download_data.sh` | Descargar dataset real (Kaggle) o caer al sintético | credenciales Kaggle (opcional) | `data/raw/fraud_train.csv` |
| `run_pipeline.py` | (Fase 5) Orquestar ETL→…→drift por `--config` | `--config` | métricas + artefacto + reporte de drift |

## `configs/`

| Archivo | Propósito |
|---|---|
| `fraud.yaml` | Parametriza el núcleo para el dominio banca/fraude (columnas, modelo, calibración, umbral, drift) |

## `tests/`

| Archivo | Cubre |
|---|---|
| `conftest.py` | Fixtures: `valid_fraud_df`, `spark` (sesión compartida) |
| `test_config.py` | Carga y validación del config |
| `test_schema.py` | Validación acepta buenos / rechaza malos datos |
| `test_validate.py` | Reporte trazable y dominios desconocidos |
| `test_ingest_spark.py` | Ingesta PySpark end-to-end → Parquet → validación |
| `test_features.py` | Features consistentes train/inferencia |
| `test_metrics.py` | PR-AUC, Brier, umbral por precisión |
| `test_calibrate.py` | La calibración mejora el Brier |
| `test_train.py` | Entrena, calibra, persiste y predice |
| `test_drift.py` | No marca datos estables; sí marca corrimiento |
| `test_api.py` | health, predict válido, rechazo de inválidos |
