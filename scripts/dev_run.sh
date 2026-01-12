#!/usr/bin/env bash
set -euo pipefail

# Cargar variables del .env (simple)
export $(grep -v '^#' .env | xargs) || true

# Usar el entorno de Poetry y asegurar PYTHONPATH
PYTHONPATH=src poetry run streamlit run src/trade_journal/app/main.py
