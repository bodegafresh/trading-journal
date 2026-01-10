# Trade Journal Pro (Supabase)

App de journal de trading con dashboard moderno (Streamlit + Plotly) y base de datos centralizada en **Supabase (Postgres)**.

## Qué incluye este repo (starter)
- Tablas `trades` y `sessions` en Supabase (SQL en `supabase/migrations/001_init.sql`)
- App Streamlit multipágina:
  - **Dashboard** (KPIs + curvas)
  - **Operaciones** (crear trade + tabla)
  - **Sesiones** (start/stop + registro)
  - **Importar** (CSV legacy -> Supabase)
- Capa de acceso a datos vía PostgREST (HTTP) para que también puedas escribir desde Apps Script / Telegram.

> Nota: este starter asume un uso “single-user”. En producción conviene activar RLS y usar auth/roles.

---

## Requisitos
- Python 3.11+

## Configuración de Supabase
1. Crea el proyecto en Supabase.
2. En el SQL editor, ejecuta el script:
   - `supabase/migrations/001_init.sql`
3. Obtén tus credenciales:
   - **Project URL**
   - **anon public key** (o service role si es solo local y privado)

Crea un `.env` a partir del ejemplo:

```bash
cp .env.example .env
```

Ejemplo:
```env
SUPABASE_URL="https://xxxx.supabase.co"
SUPABASE_KEY="tu_anon_key"
```

---

## Instalación
Con Poetry:

```bash
poetry install
poetry run streamlit run src/trade_journal/app/main.py
```

Sin Poetry (pip):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
streamlit run src/trade_journal/app/main.py
```

---

## Importar tu CSV legado
La página **Importar** acepta un CSV con las columnas del script original:

`datetime,date,asset,timeframe,amount,direction,outcome,payout_pct,pnl,emotion,notes`

---

## Estructura
- `src/trade_journal/data`: cliente Supabase + repositorios
- `src/trade_journal/domain`: modelos Pydantic y reglas
- `src/trade_journal/analytics`: KPIs + charts Plotly
- `src/trade_journal/app`: UI Streamlit (pages)

---

## Próximos pasos sugeridos
- Activar RLS + policies y dejar el bot con una key limitada
- Normalizar catálogos (assets/timeframes/emotions)
- Reportes (PDF/Markdown) y export a Sheets
