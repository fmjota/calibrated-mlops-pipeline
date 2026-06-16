# Imagen de la API de inferencia. No necesita Spark/Java (la inferencia usa LightGBM +
# sklearn), así que parte de python-slim y queda liviana.
FROM python:3.12-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# uv desde la imagen oficial (sin instalar a nivel de sistema).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) Dependencias primero, para aprovechar la cache de capas.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --no-dev --frozen

# 2) Configs del dominio.
COPY configs ./configs

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/artifacts/fraud/model.joblib

EXPOSE 8000
CMD ["uvicorn", "mlops_core.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
