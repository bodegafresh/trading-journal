# Trading Journal

Aplicación para registrar operaciones de trading (trades), gestionar sesiones y visualizar métricas con un dashboard moderno e interactivo.

> Objetivo: evolucionar un registro simple (CSV + GUI) hacia una app “pro” con base de datos, gráficos bonitos (Plotly) y flujo de trabajo mantenible.

---

## Features (MVP)

- Registro de operaciones:
  - Activo (ej: EUR/USD)
  - Timeframe
  - Monto (USD)
  - Dirección (↑ / ↓)
  - Resultado (win/loss/tie)
  - Payout %
  - Emoción
  - Notas
  - Timestamp automático
- KPIs:
  - WinRate (día y acumulado)
  - PnL diario y acumulado
  - Nº de operaciones (día y acumulado)
  - Progreso contra objetivo diario (PnL y minutos efectivos)
- Sesiones:
  - Cronómetro de sesión (inicio/pausa/finalizar)
  - Minutos efectivos por día
- Importación:
  - Importar histórico desde CSV (compatibilidad con el formato legado)
- Gráficos interactivos (Plotly):
  - PnL acumulado en el día
  - Ops por hora
  - Win/Loss por activo y timeframe
  - Distribución por emoción

---

## Tech Stack

- UI: Streamlit
- Charts: Plotly
- DB: SQLite
- Models/Validation: Pydantic
- ORM: SQLModel (o SQLAlchemy)
- Packaging: Poetry

---

## Estructura del proyecto

- `src/trade_journal/domain`: modelos y reglas de negocio (sin UI)
- `src/trade_journal/data`: persistencia (SQLite), repositorios e importadores
- `src/trade_journal/analytics`: KPIs y generación de gráficos
- `src/trade_journal/app`: aplicación Streamlit (páginas y componentes)

```md
trade-journal-pro/
├─ README.md
├─ pyproject.toml
├─ .gitignore
├─ .env.example
├─ data/
│ ├─ raw/ # imports: csv originales, backups
│ └─ app.db # sqlite local (ignorado en git)
├─ src/
│ └─ trade_journal/
│ ├─ **init**.py
│ ├─ config.py # settings (paths, env, etc.)
│ ├─ domain/
│ │ ├─ models.py # Trade, Session, enums (direction/outcome/emotion)
│ │ └─ rules.py # cálculo pnl, winrate, métricas
│ ├─ data/
│ │ ├─ database.py # engine sqlite + session
│ │ ├─ repositories.py# CRUD + queries
│ │ └─ importers.py # CSV -> DB
│ ├─ analytics/
│ │ ├─ kpis.py # KPIs diarios/acumulados
│ │ └─ charts.py # funciones Plotly
│ └─ app/
│ ├─ main.py # entrypoint Streamlit
│ ├─ pages/ # multipage dashboard
│ │ ├─ 1_Dashboard.py
│ │ ├─ 2_Operaciones.py
│ │ ├─ 3_Sesiones.py
│ │ └─ 4_Importar.py
│ └─ components.py # widgets reutilizables
├─ tests/
│ ├─ test_rules.py
│ ├─ test_kpis.py
│ └─ test_importers.py
├─ scripts/
│ ├─ dev_run.sh
│ └─ migrate_csv_to_db.py
└─ .github/
└─ workflows/
└─ ci.yml
```

---

## Instalación (desarrollo)

### Requisitos

- Python 3.11+ (recomendado)

### Setup con Poetry

```bash
poetry install
poetry run streamlit run src/trade_journal/app/main.py
```
