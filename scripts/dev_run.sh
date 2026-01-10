#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs) || true
streamlit run src/trade_journal/app/main.py
