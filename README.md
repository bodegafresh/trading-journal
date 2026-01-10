# Trade Journal Pro (Supabase)

App de journal de trading con dashboard moderno (Streamlit + Plotly) y base de datos centralizada en **Supabase (Postgres)**.

Este repo está pensado como **single-user** (sin auth). En producción conviene activar RLS + policies.

---

## Qué incluye este repo

- Base de datos Supabase (Postgres) con:
  - Tabla `trades` (operaciones)
  - Tabla `sessions` (sesiones)
- App Streamlit multipágina:
  - **Dashboard**: KPIs + curvas + segmentación por horario/timeframe/emoción **y** (nuevo) setup/régimen/calidad/checklist/sesión.
  - **Operaciones**: crear trade con campos base + campos de segmentación + sesión/evidencia.
  - **Sesiones**: start/stop + notas.
  - **Importar**: CSV legacy -> Supabase.

---

## Requisitos

- Python 3.11+

---

## Configuración de Supabase

1. Crea el proyecto en Supabase.
2. En el SQL Editor ejecuta el script de migración:
   - `supabase/migrations/001_init.sql`
3. Obtén credenciales:
   - **Project URL**
   - **anon public key** (o service role si es solo local/privado)

Crea un `.env`:

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

## Modelo de datos (Supabase)

### Tabla: `public.trades`

Registra cada operación.

#### Identidad / timestamps

- `id` (uuid, PK): id único.
- `trade_time` (timestamptz): timestamp del trade. **Recomendado: UTC**.
- `trade_date` (date): fecha derivada por trigger desde `trade_time` en UTC.
- `created_at` (timestamptz): creación del registro.

#### Datos core del trade

- `asset` (text): activo (ej: `EUR/USD`, `BTC/USD`).
- `timeframe` (text): timeframe (ej: `1m`, `5m`).
- `amount` (numeric): stake/monto en USD.
- `direction` (text): `UP` / `DOWN`.
- `outcome` (text): `WIN` / `LOSS` / `TIE`.
- `payout_pct` (numeric): payout en % (ej: 80).
- `pnl` (numeric): PnL en USD.
- `emotion` (text): emoción (catálogo).
- `notes` (text, nullable): notas libres.

#### Campos nuevos (segmentación / calidad / contexto)

- `setup_tag` (text, nullable)
- `market_regime` (text, nullable)
- `quality_grade` (text, nullable)
- `checklist_passed` (bool)
- `session_id` (uuid, nullable, FK a `sessions.id`)
- `screenshot_url` (text, nullable)

> **Regla de oro**: lo que más vale no es “tener el campo”, sino **ser consistente** al completarlo.

---

### Tabla: `public.sessions`

Permite registrar sesiones de trading.

- `id` (uuid, PK)
- `start_time` (timestamptz): inicio
- `end_time` (timestamptz, nullable): fin
- `duration_min` (numeric, nullable): duración (si la calculas)
- `notes` (text, nullable)
- `created_at` (timestamptz)

---

## Operaciones (cómo completar cada campo)

### 1) Campos base (core)

- **Activo (`asset`)**: selecciona el instrumento.
- **Timeframe (`timeframe`)**: 1m / 5m (o los que definas).
- **Monto (`amount`)**: stake en USD.
- **Payout % (`payout_pct`)**: porcentaje de retorno de la binaria.
- **Dirección (`direction`)**: UP/DOWN.
- **Resultado (`outcome`)**: WIN/LOSS/TIE.
- **Emoción (`emotion`)**: emoción dominante antes/durante la entrada.
- **Notas (`notes`)**: contexto textual.

#### Fórmulas base (binarias)

- Si `outcome = WIN`:
  [
  PnL = Amount \cdot \frac{payout_pct}{100}
  ]
- Si `outcome = LOSS`:
  [
  PnL = -Amount
  ]
- Si `outcome = TIE`:
  [
  PnL = 0
  ]

**R-multiple (R)**
En el dashboard se usa un “R” práctico:
[
R = \frac{PnL}{Amount}
]

- WIN con payout 80% ⇒ (R = +0.8)
- LOSS ⇒ (R = -1)
- TIE ⇒ (R = 0)

**EV en R (promedio)**
[
EV_R = \mathbb{E}[R]
]
En datos reales, el dashboard lo calcula como el promedio de `R` (r_mult).

---

## Campos nuevos (explicados en detalle)

### 1) `setup_tag` — “Setup”

**Qué es:** el patrón/estrategia que estabas ejecutando. Es tu etiqueta principal para analizar “qué setup tiene mejor EV”.

**Valores posibles (según tu UI):**

- (vacío) → significa **None / sin clasificar**
- `Breakout`
- `Reversal`
- `Trend`
- `Range`
- `News`

**Cómo lo completo:**

- Elige uno que represente la **idea del trade**.
- Si no estás seguro, **déjalo vacío** (mejor vacío que mal etiquetado).

**Ejemplos rápidos:**

- Rompió una zona y continuó → `Breakout`
- Se devolvió desde resistencia/soporte → `Reversal`
- Entraste a favor de tendencia clara → `Trend`
- Mercado lateral entre techo/piso → `Range`
- Entrada por noticia/evento → `News`

---

### 2) `market_regime` — “Régimen de mercado”

**Qué es:** el contexto del mercado en ese momento. Te sirve para responder:

> “¿Este setup funciona igual en tendencia que en rango?”

**Valores posibles:**

- (vacío) → None
- `Trend`
- `Range`
- `Volatile`
- `LowVol`

**Cómo lo completo:**
No es “tu setup”, es el **clima** del mercado. Usa reglas simples y consistentes.

**Guía práctica:**

- `Trend`: velas direccionales, HH/HL o LL/LH, empuje claro.
- `Range`: ida y vuelta en un canal/zona, rechazos repetidos.
- `Volatile`: velas grandes, mechas fuertes, movimientos bruscos/spikes.
- `LowVol`: velas pequeñas, poco rango, mercado “apretado”.

---

### 3) `quality_grade` — “Calidad”

**Qué es:** tu nota de ejecución (qué tan “de manual” fue el trade).

**Valores posibles:**

- (vacío) → None
- `A`, `B`, `C`, `D`

**Cómo lo completo (criterio recomendado):**

- `A`: totalmente según plan (setup claro + confirmación + timing + gestión).
- `B`: bueno, pero con 1 detalle mejorable (entrada algo tarde, confirmación parcial, etc.).
- `C`: ejecutaste algo “medio”, faltó claridad o paciencia.
- `D`: impulsivo / fuera de plan / mala calidad.

**Por qué es importante:**
Te permite medir:

- EV de trades `A` vs `B` vs `C` vs `D`
- Si `A` realmente paga y `D` te hunde (muy común).

---

### 4) `checklist_passed` — “Checklist PASS”

**Qué es:** booleano (Sí/No) que marca si cumpliste tu checklist antes de ejecutar.

**Valores posibles:**

- `true` → PASS
- `false` → FAIL

**Cómo lo completo:**

- Si antes de entrar cumpliste los puntos mínimos → PASS
- Si entraste sin cumplirlos → FAIL

**Por qué es importante:**
Es la forma más directa de comprobar si “ser disciplinado” aumenta el EV:

- filtro: “solo PASS”
- comparación: PASS vs FAIL

> Nota: En tu UI aparece “Checklist PASS”. Si no lo marcaste, queda en `false` (FAIL).

---

### 5) `session_id` — “Sesión”

**Qué es:** link a una sesión de trading (tabla `sessions`). Sirve para agrupar trades por sesión:

> “Esta sesión fue buena/mala”.

**Valores posibles:**

- `None` → “(Sin sesión)”
- Un `uuid` que apunta a `public.sessions.id`

**Cómo lo completo:**

- Si el trade fue dentro de una sesión registrada → seleccionas esa sesión en el dropdown.
- Si no estás usando sesiones todavía → déjalo “(Sin sesión)”.

**Qué podrás analizar después:**

- EV por sesión
- PnL por sesión
- Sesiones con peor drawdown
- “Cuándo me conviene cortar una sesión”

---

### 6) `screenshot_url` — “Screenshot/Link evidencia”

**Qué es:** una URL a evidencia: imagen, link de TradingView, Drive, etc.

**Valores posibles:**

- `None` (vacío)
- texto con URL (string)

**Cómo lo completo:**

- Pegas un link (idealmente algo estable: snapshot de TradingView, link compartido, etc.)
- Si no hay evidencia, lo dejas vacío.

**Para qué sirve:**
Auditoría rápida: revisar el contexto visual del trade.
Ideal para revisar trades `D` o `FAIL` y corregir patrones.

---

## Cómo llenarlo bien (regla simple)

Si tu objetivo es mejorar rápido:

- Siempre marca **Checklist PASS/FAIL** con honestidad.
- Siempre etiqueta **Setup** cuando estés seguro.
- Usa **Market Regime** y **Quality** para entender por qué el setup funcionó o no.
- Evidence es opcional, pero muy útil en trades problemáticos.

---

## Ejemplo de un trade completo

- Setup: `Breakout`
- Régimen: `Trend`
- Calidad: `A`
- Checklist: `PASS`
- Sesión: “2026-01-10 AM”
- Evidencia: link a TradingView
- Notas: “Rompió rango + retest, entré en confirmación.”

---

## Dashboard (qué muestra y cómo se calcula)

### Resumen del día (fecha local)

Filtra por `trade_date_local` (derivada desde `trade_time` en zona local) y muestra:

- Trades
- W/L/T
- WinRate (incl. tie):
  [
  WR_{incl}=\frac{W}{W+L+T}
  ]
- WinRate (sin tie / decided only):
  [
  WR_{decided}=\frac{W}{W+L}
  ]
- PnL del día
- EV (R):
  [
  EV_R=\mathbb{E}\left[\frac{PnL}{Amount}\right]
  ]

### Acumulado

Lo mismo pero sobre todo el histórico cargado.

### Equity en R & Drawdown

- (equity_R) = cumsum de (R)
- (peak) = máximo acumulado de (equity_R)
- (DD = equity_R - peak) (siempre ≤ 0)

### Segmentaciones (tablas + gráficos)

Además de horario/timeframe/emoción, con los campos nuevos puedes ver:

- EV/WinRate por `setup_tag`
- EV/WinRate por `market_regime`
- EV/WinRate por `quality_grade`
- PASS vs FAIL (`checklist_passed`)
- (Opcional) por `session_id` si lo usas

> Objetivo: convertir el dashboard en un “mapa” de dónde está tu EV (qué te paga y qué te drena).

---

## Importar CSV legado

La página **Importar** acepta el CSV del script antiguo:

`datetime,date,asset,timeframe,amount,direction,outcome,payout_pct,pnl,emotion,notes`

Notas:

- `datetime` se parsea a `trade_time`.
- Los campos nuevos pueden quedar en NULL si el CSV no los trae.

---

## Estructura del repo

- `src/trade_journal/data`: cliente Supabase + repositorios
- `src/trade_journal/domain`: modelos Pydantic y reglas
- `src/trade_journal/analytics`: KPIs + charts Plotly
- `src/trade_journal/app`: UI Streamlit (pages)

---

## Próximos pasos sugeridos

- Activar RLS + policies (si lo vas a exponer).
- Normalizar catálogos (assets/timeframes/emotions/setups/regimes/grades).
- Export/Reportes (Markdown/PDF) y export a Sheets.
- Filtros globales en dashboard (por activo, setup, sesión, rango de fechas).
