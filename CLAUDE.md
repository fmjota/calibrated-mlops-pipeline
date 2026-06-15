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
- **2026-06-15** — **PySpark + JDK 17**. La notebook tiene Java 25, que Spark aún no
  soporta; se usará un JDK 17 dedicado con `JAVA_HOME` apuntado a él (sin cambiar el
  Java por defecto del sistema).
- **2026-06-15** — Dataset propuesto: *Credit Card Transactions Fraud Detection*
  (Sparkov, ~1.85M filas). Alternativa liviana: ULB `creditcard.csv` (285k, PCA).

## Estado actual

Fase 0 (scaffolding + entorno) en curso. Plan completo por fases en la raíz del
portafolio (`prompt-portafolio-proyectos-github.md`) y en el plan aprobado de la sesión.
