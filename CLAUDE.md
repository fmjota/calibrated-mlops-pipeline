# CLAUDE.md — Proyecto 1: Pipeline de MLOps de extremo a extremo

Memoria de trabajo del repo entre sesiones. Léelo antes de retomar.

## Qué es esto

Primer proyecto de un portafolio de 3 (datos + IA) para defender en entrevistas. Es el
**ancla de credibilidad**: lleva un modelo de datos crudos a producción de forma
**validada, calibrada y monitoreada**. El núcleo se construye **una vez** y se demuestra
sobre tres dominios cambiando solo `config` + esquema (banca → salud → educación), no
reescribiendo el pipeline.

**Hilo conductor (ventaja estadística del autor):** validación temprana y trazable,
**probabilidades calibradas** (no solo rankeadas), **incertidumbre** (variante bayesiana
donde aporta) y **monitoreo de drift** como detección de señales aplicada al modelo.

## Stack y por qué

| Pieza | Elección | Por qué |
|---|---|---|
| Gestor de entorno | **uv** | Rápido, lockfile reproducible, no toca el Python del sistema |
| Ingesta/ETL | **PySpark** | ETL distribuido; el dataset de fraude (~1.85M filas) lo justifica |
| Validación | **Pandera** | Fallar temprano: contrato de datos explícito antes del modelo |
| Modelo base | **LightGBM** | En tabular con desbalance suele ganar a deep learning en performance/costo/explicabilidad |
| Calibración | **isotónica / Platt** | El umbral de decisión debe significar algo; se elige por **Brier** |
| Tracking | **MLflow** | Params, métricas (PR-AUC, Brier) y artefactos reproducibles |
| Drift | **PSI / KS** | Entrenamiento vs producción; detecta degradación silenciosa |
| Serving | **FastAPI + Docker** | API de inferencia que devuelve probabilidad calibrada + decisión |

Métricas honestas para desbalance: **PR-AUC** y **Brier**, no accuracy.

## Estructura

- `src/mlops_core/` — núcleo agnóstico al dominio (config, schemas, ingest, validate,
  features, models, evaluate, drift, serve).
- `configs/*.yaml` — un archivo por dominio; parametriza el núcleo.
- `tests/` — pytest por componente.
- `scripts/` — `download_data.sh`, `run_pipeline.py` (orquesta el end-to-end por `--config`).

## Comandos

```bash
uv sync                                   # crea .venv e instala deps (Python 3.12)
uv run pytest                             # tests
uv run ruff check . && uv run ruff format # lint + formato
uv run pre-commit install                 # activa hooks de calidad
uv run python scripts/run_pipeline.py --config configs/fraud.yaml
```

## Convenciones

- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `chore:`). Historial limpio.
- Construir **incremental**, con tests verdes en cada paso.
- **Explicar cada decisión técnica**; si se toma un atajo, decirlo.
- **Nunca** instalar en el Python del sistema; usar uv. **PySpark requiere JDK**.
- Pedir OK antes de decisiones grandes (dataset, cambiar librería del stack) o de
  instalar a nivel de sistema.

## Bitácora de decisiones

- **2026-06-15** — Dominio inicial: **banca/fraude de tarjetas** (caso canónico que un
  reclutador sabe evaluar; luce calibración + PR-AUC de inmediato). Salud y educación
  vienen después sobre el mismo núcleo.
- **2026-06-15** — Entorno **uv** (no venv plano). Repos: **una subcarpeta por proyecto**
  del portafolio; este es `proyecto-1-mlops/`.
- **2026-06-15** — **PySpark 4.1 + Temurin 21 (LTS) a nivel de usuario**. Fedora 44 solo
  trae Java 25/26 en repos (no 17/21), y Spark 4.x no soporta Java 25. Se instaló un JDK
  21 portable en `~/.local/share/jvm/jdk-21*` (sin sudo, sin tocar el sistema).
  `mlops_core.spark.get_spark()` descubre ese JDK y fija `JAVA_HOME` solo para Spark.
- **2026-06-15** — Datos: el ETL apunta al dataset **Sparkov** real (`download_data.sh`
  vía Kaggle si hay credenciales). Como fallback portable —tests/CI/demo sin credenciales—
  `scripts/generate_synthetic.py` genera datos con el **mismo esquema** (~0.13% fraude,
  desbalance realista). El núcleo no distingue entre ambos.

- **2026-06-15** — `target_precision` es un **knob de negocio** en el config, no un valor
  sagrado. En el dataset sintético (0.6% fraude) con 0.90 el umbral quedaba en 1.0 (recall
  ~2%); se fijó en **0.50**, que da un punto de operación ilustrativo (umbral ~0.42,
  precisión ~0.67, recall ~0.22). La calibración elige isotónica vs Platt por Brier.

## Resultados de referencia (dataset sintético 200k, 0.6% fraude)

ROC-AUC ~0.98, PR-AUC calibrado ~0.37 (≈60× sobre azar), Brier 0.0064 → **0.0043** con
calibración isotónica. Punto de operación @ precisión objetivo 0.50: umbral ~0.42,
precisión ~0.67, recall ~0.22. (Cambiarán con el dataset real de Kaggle.)

## Estado actual

Fases 0-3 completas. **Fase 0:** entorno uv, scaffolding, CI, pre-commit.
**Fase 1:** ingesta PySpark CSV→Parquet + validación Pandera trazable. **Fase 2:** features
agnósticas al dominio (fecha + distancia haversine), LightGBM con desbalance, calibración
isotónica/Platt por Brier, métricas honestas (PR-AUC/Brier) y tracking MLflow; modelo
calibrado serializado para serving. **Fase 3:** detección de drift por feature (PSI + KS)
referencia vs producción, con umbrales del config. **31 tests verdes.** Próximo: **Fase 4**
(serving FastAPI + Docker).
