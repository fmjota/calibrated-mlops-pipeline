#!/usr/bin/env bash
# Descarga el dataset de fraude. Usa el dataset real de Kaggle si hay credenciales;
# si no, genera uno sintético equivalente (mismo esquema) para que el pipeline corra igual.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw data/processed
TARGET="data/raw/fraud_train.csv"

if [ -f "$TARGET" ]; then
  echo "Ya existe $TARGET (borra el archivo para regenerarlo)."
  exit 0
fi

# 1) Dataset real de Kaggle (requiere ~/.kaggle/kaggle.json).
if command -v kaggle >/dev/null 2>&1; then
  echo "Intentando descargar dataset real desde Kaggle (kartik2112/fraud-detection)..."
  if kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip; then
    [ -f data/raw/fraudTrain.csv ] && mv data/raw/fraudTrain.csv "$TARGET"
  fi
fi

# 2) Fallback sintético (corre en cualquier lado y en CI).
if [ ! -f "$TARGET" ]; then
  echo "Sin Kaggle disponible: generando dataset sintético equivalente..."
  uv run python scripts/generate_synthetic.py --rows 200000 --out "$TARGET"
fi

echo "Listo: $TARGET"
