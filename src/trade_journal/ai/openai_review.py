from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import tiktoken
from openai import OpenAI


# ----------------------------
# Prompt v1.0 definitivo (sesión)
# ----------------------------
PROMPT_REVIEW_SESSION_V1 = r"""
Eres un auditor cuantitativo y coach de ejecución A+ (estricto, sin humo).
NO das señales de entrada ni propones estrategias nuevas.
Tu rol es auditar si el trade/sesión fue A+ real, detectar fallas de ejecución y proponer reglas accionables.

Contexto:
- Estrategia: seguimiento de tendencia intradía, continuación tras pullback
- Timeframe: 1m
- Alta selectividad (solo setups A+)
- Disciplina como métrica principal
- Indicadores: EMA 21, EMA 6, EMA 3, Bollinger(20,2), RSI(7) como filtro de “zona muerta” (45–55)

Checklist oficial A+ (5 bloques; TODOS deben cumplirse):
1) context:
   - Mercado NO lateral
   - EMA21 con pendiente clara
   - Precio claro a un lado de EMA21
   - Bollinger abierta (no comprimida)
2) ema21:
   - Precio arriba EMA21 -> solo CALL
   - Precio abajo EMA21 -> solo PUT
   - EMA21 no plana
3) ema3_6:
   - EMA3 y EMA6 alineadas con la dirección
   - No cruces caóticos recientes
   - Secuencia impulso → pausa → continuación (pullback ordenado)
4) entry_candle:
   - Vela con cuerpo claro
   - Sin mechas largas en contra
   - Entrada NO tardía (no perseguir vela / no entrar en extensión)
5) no_entry_filters:
   - No aburrimiento / urgencia
   - No operar para recuperar
   - No operar tras 3 pérdidas consecutivas
   - Operar solo en horario definido

REGLAS DURAS (no negociables):
- Si `context` = fail OR `no_entry_filters` = fail => `ai_validity` DEBE ser false.
- “A+ real” es estricto:
  - `ai_is_a_plus` = true SOLO si ai_score_a_plus_0_5 == 5 Y ai_validity == true.
- trade_type (clasificación por outcome y validez):
  - Si outcome == LOSS y ai_validity == true => good_loss (pérdida válida, setup correcto)
  - Si outcome == LOSS y ai_validity == false => bad_loss (error de ejecución)
  - Si outcome == WIN y ai_validity == true => good_win (ganancia válida, replicable)
  - Si outcome == WIN y ai_validity == false => lucky_win (ganancia por suerte, no replicable)
  - Si outcome == TIE => neutral_tie
- Prohibido afirmar “pasó checklist” si ai_validity=false.
- Nada de consejos genéricos: cada corrección debe ser una acción concreta aplicable en el próximo trade.

Input:
Recibirás JSON con:
- session_meta (incluye scope: "session")
- kpis
- trades_sample
- trades_with_images (TODAS las operaciones con imagen: WIN, LOSS, TIE)

Tu tarea:
1) Entregar resumen SIN contradicciones, basado en datos (resumen de la sesión).
2) Auditar CADA operación con evidencia (imagen) y llenar checklist por bloques.
3) Para operaciones con fallas: listar failed_blocks y failed_rules (reglas específicas dentro de cada bloque).
4) PROFUNDIZAR en las PÉRDIDAS: analizar qué salió mal, qué resolver, qué mejorar.
5) Analizar GANANCIAS: identificar si fueron por ejecución correcta (replicable) o suerte.
6) Analizar EMPATES: qué pudo haberse mejorado en la ejecución.
7) Producir reglas accionables de la sesión (máx 5, concretas), priorizando lo que previene pérdidas y maximiza ganancias replicables.

FORMATO DE SALIDA (OBLIGATORIO):
- Devuelve SOLO JSON válido. Nada de texto fuera del JSON.
- No uses comillas simples. No agregues comentarios. No uses trailing commas.
- El JSON debe ser parseable por json.loads en Python.
Estructura:

{
  "session_summary_md": "markdown breve (máx ~30 líneas) con resumen de sesión, patrones en pérdidas, ganancias y empates",
  "checklist_findings": {
    "pass_vs_fail_interpretation": "texto corto sobre adherencia al protocolo",
    "most_common_breaks": ["...", "..."],
    "win_analysis": "patrón común en las ganancias (¿setup correcto o suerte?)",
    "loss_analysis": "patrón común en las pérdidas (¿qué falló más?)",
    "tie_analysis": "patrón en empates si existen"
  },
  "action_rules": ["...", "...", "...", "...", "..."],
  "trade_reviews": [
    {
      "trade_id": "uuid",
      "outcome": "WIN|LOSS|TIE",
      "ai_score_a_plus_0_5": 0,
      "ai_validity": true,
      "ai_is_a_plus": false,
      "trade_type": "good_loss|bad_loss|good_win|lucky_win|neutral_tie",
      "checklist": {
        "context": "pass|fail|unclear",
        "ema21": "pass|fail|unclear",
        "ema3_6": "pass|fail|unclear",
        "entry_candle": "pass|fail|unclear",
        "no_entry_filters": "pass|fail|unclear"
      },
      "failed_blocks": ["context", "entry_candle"],
      "failed_rules": ["Bollinger comprimida", "Entrada tardía (persecución)"],
      "primary_cause": "frase corta sobre la causa raíz",
      "what_to_fix": "qué resolver específicamente",
      "what_to_improve": "qué mejorar en la ejecución",
      "key_lesson": "lección clave de esta operación (especialmente importante en pérdidas)",
      "replicability": "si WIN: ¿es replicable este resultado? / si LOSS: ¿era evitable?",
      "confidence_0_1": 0.0
    }
  ]
}

NOTA IMPORTANTE: Para PÉRDIDAS, profundiza más en 'what_to_fix', 'what_to_improve' y 'key_lesson'.
Para GANANCIAS, enfócate en 'replicability' y si el setup fue genuinamente A+.
Para EMPATES, indica qué pudo haberse ejecutado mejor para convertirlo en ganancia.
"""

# ----------------------------
# Prompt v1.0 definitivo (semanal)
# ----------------------------
PROMPT_REVIEW_WEEKLY_V1 = r"""
Eres un auditor cuantitativo y coach de ejecución A+ (estricto, sin humo).
NO das señales de entrada ni propones estrategias nuevas.
Tu rol es auditar si el conjunto de trades de la semana fue A+ real, detectar fallas de ejecución y proponer reglas accionables.

Contexto:
- Estrategia: seguimiento de tendencia intradía, continuación tras pullback
- Timeframe: 1m
- Alta selectividad (solo setups A+)
- Disciplina como métrica principal
- Indicadores: EMA 21, EMA 6, EMA 3, Bollinger(20,2), RSI(7) como filtro de “zona muerta” (45–55)

Checklist oficial A+ (5 bloques; TODOS deben cumplirse):
1) context:
   - Mercado NO lateral
   - EMA21 con pendiente clara
   - Precio claro a un lado de EMA21
   - Bollinger abierta (no comprimida)
2) ema21:
   - Precio arriba EMA21 -> solo CALL
   - Precio abajo EMA21 -> solo PUT
   - EMA21 no plana
3) ema3_6:
   - EMA3 y EMA6 alineadas con la dirección
   - No cruces caóticos recientes
   - Secuencia impulso → pausa → continuación (pullback ordenado)
4) entry_candle:
   - Vela con cuerpo claro
   - Sin mechas largas en contra
   - Entrada NO tardía (no perseguir vela / no entrar en extensión)
5) no_entry_filters:
   - No aburrimiento / urgencia
   - No operar para recuperar
   - No operar tras 3 pérdidas consecutivas
   - Operar solo en horario definido

REGLAS DURAS (no negociables):
- Si `context` = fail OR `no_entry_filters` = fail => `ai_validity` DEBE ser false.
- “A+ real” es estricto:
  - `ai_is_a_plus` = true SOLO si ai_score_a_plus_0_5 == 5 Y ai_validity == true.
- trade_type (clasificación por outcome y validez):
  - Si outcome == LOSS y ai_validity == true => good_loss (pérdida válida, setup correcto)
  - Si outcome == LOSS y ai_validity == false => bad_loss (error de ejecución)
  - Si outcome == WIN y ai_validity == true => good_win (ganancia válida, replicable)
  - Si outcome == WIN y ai_validity == false => lucky_win (ganancia por suerte, no replicable)
  - Si outcome == TIE => neutral_tie
- Prohibido afirmar “pasó checklist” si ai_validity=false.
- Nada de consejos genéricos: cada corrección debe ser una acción concreta aplicable en el próximo trade.

Input:
Recibirás JSON con:
- session_meta (incluye scope: "weekly")
- kpis
- kpis_by_asset (opcional): rendimiento agregado por activo
- kpis_by_hour (opcional): rendimiento agregado por hora
- trades_sample
- trades_with_images (TODAS las operaciones con imagen: WIN, LOSS, TIE)
- session_reviews (opcional): reviews previas por sesión para usar como contexto semanal
- session_trade_reviews_summary (opcional): resumen agregado de trade_reviews de sesiones
- previous_week_review (opcional): resumen/insights de la semana anterior para comparar evolución

Tu tarea:
1) Entregar resumen semanal SIN contradicciones, basado en datos (usa la palabra "semana").
2) Explicar evolución de la semana: qué mejoró/qué empeoró sobre cuándo entrar y cuándo no entrar.
3) Comparar con la semana anterior si se entrega previous_week_review.
4) Auditar CADA operación con evidencia (imagen) y llenar checklist por bloques.
5) Para operaciones con fallas: listar failed_blocks y failed_rules (reglas específicas dentro de cada bloque).
6) PROFUNDIZAR en las PÉRDIDAS: analizar qué salió mal, qué resolver, qué mejorar.
7) Analizar GANANCIAS: identificar si fueron por ejecución correcta (replicable) o suerte.
8) Analizar EMPATES: qué pudo haberse mejorado en la ejecución.
9) Usar el contexto de session_reviews y session_trade_reviews_summary para dar mejor feedback.
10) Producir reglas accionables de la semana (máx 5, concretas), priorizando lo que previene pérdidas y maximiza ganancias replicables.

FORMATO DE SALIDA (OBLIGATORIO):
- Devuelve SOLO JSON válido. Nada de texto fuera del JSON.
- No uses comillas simples. No agregues comentarios. No uses trailing commas.
- El JSON debe ser parseable por json.loads en Python.
Estructura:

{
  "session_summary_md": "markdown breve (máx ~30 líneas) con resumen de la semana, patrones en pérdidas, ganancias y empates",
  "checklist_findings": {
    "pass_vs_fail_interpretation": "texto corto sobre adherencia al protocolo",
    "most_common_breaks": ["...", "..."],
    "win_analysis": "patrón común en las ganancias (¿setup correcto o suerte?)",
    "loss_analysis": "patrón común en las pérdidas (¿qué falló más?)",
    "tie_analysis": "patrón en empates si existen"
  },
  "action_rules": ["...", "...", "...", "...", "..."],
  "trade_reviews": [
    {
      "trade_id": "uuid",
      "outcome": "WIN|LOSS|TIE",
      "ai_score_a_plus_0_5": 0,
      "ai_validity": true,
      "ai_is_a_plus": false,
      "trade_type": "good_loss|bad_loss|good_win|lucky_win|neutral_tie",
      "checklist": {
        "context": "pass|fail|unclear",
        "ema21": "pass|fail|unclear",
        "ema3_6": "pass|fail|unclear",
        "entry_candle": "pass|fail|unclear",
        "no_entry_filters": "pass|fail|unclear"
      },
      "failed_blocks": ["context", "entry_candle"],
      "failed_rules": ["Bollinger comprimida", "Entrada tardía (persecución)"],
      "primary_cause": "frase corta sobre la causa raíz",
      "what_to_fix": "qué resolver específicamente",
      "what_to_improve": "qué mejorar en la ejecución",
      "key_lesson": "lección clave de esta operación (especialmente importante en pérdidas)",
      "replicability": "si WIN: ¿es replicable este resultado? / si LOSS: ¿era evitable?",
      "confidence_0_1": 0.0
    }
  ]
}

NOTA IMPORTANTE: Para PÉRDIDAS, profundiza más en 'what_to_fix', 'what_to_improve' y 'key_lesson'.
Para GANANCIAS, enfócate en 'replicability' y si el setup fue genuinamente A+.
Para EMPATES, indica qué pudo haberse ejecutado mejor para convertirlo en ganancia.
"""


@dataclass
class TradeLite:
    id: str
    trade_time: str
    asset: str
    timeframe: str
    direction: str
    outcome: str
    amount: float
    payout_pct: float
    pnl: float
    r_mult: float
    emotion: str
    checklist_pass: Optional[bool]
    screenshot_url: Optional[str]
    notes: Optional[str]


def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno (.env).")
    return OpenAI(api_key=key)


def _model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"


def _is_image_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower().split("?")[0]
    return u.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _to_jsonable(x: Any) -> Any:
    """
    Convierte tipos problemáticos (Timestamp, datetime, numpy, etc.)
    a formatos JSON seguros.
    """
    if x is None:
        return None
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    iso = getattr(x, "isoformat", None)
    if callable(iso):
        try:
            return x.isoformat()
        except Exception:
            pass
    try:
        import numpy as np  # noqa: F401

        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass
    if isinstance(x, set):
        return list(x)
    return x


def _json_dumps_safe(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_to_jsonable)


def _hash_payload(payload_text: str) -> str:
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16]


def _count_text_tokens(model: str, text: str) -> int:
    """
    Estimación SOLO texto (tiktoken).
    Ojo: imágenes no se contabilizan aquí.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _usage_to_dict(resp: Any) -> Dict[str, Any]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    d: Dict[str, Any] = {}
    for k in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"):
        v = getattr(usage, k, None)
        if v is not None:
            d[k] = v
    return d


def build_session_payload(
    *,
    session_meta: Dict[str, Any],
    trades: List[TradeLite],
    max_trades_with_images: int = 20,
    prioritize_losses: bool = True,
) -> Dict[str, Any]:
    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]
    ties = [t for t in trades if t.outcome == "TIE"]

    # Recolectar todas las operaciones con imágenes
    trades_with_img = [
        t for t in trades
        if t.screenshot_url and _is_image_url(t.screenshot_url)
    ]

    # Si se prioriza pérdidas, ordenar para que aparezcan primero
    if prioritize_losses:
        losses_imgs = [t for t in trades_with_img if t.outcome == "LOSS"]
        wins_imgs = [t for t in trades_with_img if t.outcome == "WIN"]
        ties_imgs = [t for t in trades_with_img if t.outcome == "TIE"]
        # Prioridad: todas las pérdidas primero, luego wins, luego ties
        trades_with_img = losses_imgs + wins_imgs + ties_imgs

    # Limitar al máximo especificado
    trades_with_img = trades_with_img[:max_trades_with_images]

    chk_pass = sum(1 for t in trades if t.checklist_pass is True)
    chk_fail = sum(1 for t in trades if t.checklist_pass is False)
    chk_none = sum(1 for t in trades if t.checklist_pass is None)

    total = len(trades)
    wr_with_ties = (len(wins) / total) if total else 0.0
    decided = len(wins) + len(losses)
    wr_no_ties = (len(wins) / decided) if decided else 0.0
    tie_rate = (len(ties) / total) if total else 0.0
    pnl_total = sum(t.pnl for t in trades)
    ev_r = (sum(t.r_mult for t in trades) / total) if total else 0.0

    session_meta_safe = {k: _to_jsonable(v) for k, v in (session_meta or {}).items()}

    return {
        "session_meta": session_meta_safe,
        "kpis": {
            "trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "ties": len(ties),
            "wr_with_ties": wr_with_ties,
            "wr_no_ties": wr_no_ties,
            "tie_rate": tie_rate,
            "pnl_total": pnl_total,
            "ev_r": ev_r,
            "checklist_user_declared": {
                "pass": chk_pass,
                "fail": chk_fail,
                "missing": chk_none,
            },
            "trades_with_images_used": len(trades_with_img),
            "trades_with_images_by_outcome": {
                "losses": len([t for t in trades_with_img if t.outcome == "LOSS"]),
                "wins": len([t for t in trades_with_img if t.outcome == "WIN"]),
                "ties": len([t for t in trades_with_img if t.outcome == "TIE"]),
            },
        },
        "trades_sample": [
            {
                "id": t.id,
                "trade_time": _safe_str(t.trade_time),
                "asset": t.asset,
                "timeframe": t.timeframe,
                "direction": t.direction,
                "outcome": t.outcome,
                "amount": t.amount,
                "payout_pct": t.payout_pct,
                "pnl": t.pnl,
                "r_mult": t.r_mult,
                "emotion": t.emotion,
                "checklist_pass": t.checklist_pass,
                "screenshot_url": t.screenshot_url,
                "notes": t.notes,
            }
            for t in trades[-80:]
        ],
        "trades_with_images": [
            {
                "id": t.id,
                "trade_time": _safe_str(t.trade_time),
                "asset": t.asset,
                "timeframe": t.timeframe,
                "direction": t.direction,
                "outcome": t.outcome,
                "amount": t.amount,
                "payout_pct": t.payout_pct,
                "pnl": t.pnl,
                "r_mult": t.r_mult,
                "emotion": t.emotion,
                "checklist_pass": t.checklist_pass,
                "screenshot_url": t.screenshot_url,
                "notes": t.notes,
            }
            for t in trades_with_img
        ],
    }


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        # fallback: intentar recortar entre llaves más externas
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_review(review: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(review or {})

    # Compatibilidad: algunos modelos devuelven campos planos
    if "checklist_findings" not in out:
        if any(k in out for k in ("pass_vs_fail_interpretation", "most_common_breaks", "win_analysis", "loss_analysis", "tie_analysis")):
            out["checklist_findings"] = {
                "pass_vs_fail_interpretation": out.get("pass_vs_fail_interpretation", "—"),
                "most_common_breaks": out.get("most_common_breaks") or [],
                "win_analysis": out.get("win_analysis", "—"),
                "loss_analysis": out.get("loss_analysis", "—"),
                "tie_analysis": out.get("tie_analysis", "—"),
            }
    if "action_rules" not in out:
        out["action_rules"] = []
    summary = out.get("session_summary_md")
    if summary is None or str(summary).strip() == "":
        out["session_summary_md"] = "—"

    # Completar faltantes en checklist_findings si existe
    if "checklist_findings" in out:
        cf = out.get("checklist_findings") or {}
        out["checklist_findings"] = {
            "pass_vs_fail_interpretation": cf.get("pass_vs_fail_interpretation") or "—",
            "most_common_breaks": cf.get("most_common_breaks") or [],
            "win_analysis": cf.get("win_analysis") or "—",
            "loss_analysis": cf.get("loss_analysis") or "—",
            "tie_analysis": cf.get("tie_analysis") or "—",
        }

    # Fallback mínimo de action_rules si la IA no responde
    if isinstance(out.get("action_rules"), list) and len(out["action_rules"]) == 0:
        cf = out.get("checklist_findings") or {}
        breaks = set(cf.get("most_common_breaks") or [])
        rules: List[str] = []
        if "context" in breaks:
            rules.append("No operar si el mercado está lateral o con Bollinger comprimida.")
        if "ema21" in breaks:
            rules.append("Operar solo a favor de la EMA21 con pendiente clara.")
        if "ema3_6" in breaks:
            rules.append("Exigir alineación limpia EMA3/EMA6 y evitar cruces recientes.")
        if "entry_candle" in breaks:
            rules.append("Evitar entradas tardías y velas con mecha en contra.")
        if "no_entry_filters" in breaks:
            rules.append("No operar por urgencia ni tras 3 pérdidas consecutivas.")
        if len(rules) < 3:
            rules.extend(
                [
                    "Esperar un pullback ordenado antes de entrar.",
                    "Reducir operaciones fuera del horario definido.",
                    "Priorizar activos con mejor EV semanal.",
                ]
            )
        out["action_rules"] = rules[:5]

    return _enforce_hard_rules(out)


def _enforce_hard_rules(review: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(review or {})
    # Soportar ambos nombres por compatibilidad
    ltr = out.get("trade_reviews") or out.get("loss_trade_reviews") or []
    fixed: List[Dict[str, Any]] = []

    for item in ltr:
        d = dict(item or {})
        checklist = d.get("checklist") or {}
        context = str(checklist.get("context", "unclear")).lower()
        no_entry = str(checklist.get("no_entry_filters", "unclear")).lower()

        ai_validity = bool(d.get("ai_validity", True))
        if context == "fail" or no_entry == "fail":
            ai_validity = False
        d["ai_validity"] = ai_validity

        score = d.get("ai_score_a_plus_0_5", 0)
        try:
            score = int(score)
        except Exception:
            score = 0
        score = max(0, min(5, score))
        d["ai_score_a_plus_0_5"] = score

        d["ai_is_a_plus"] = bool(ai_validity and score == 5)

        # Clasificación por tipo de trade
        outcome = str(d.get("outcome", "LOSS")).upper()
        trade_type = d.get("trade_type")
        if not trade_type:
            if outcome == "LOSS":
                trade_type = "good_loss" if ai_validity else "bad_loss"
            elif outcome == "WIN":
                trade_type = "good_win" if ai_validity else "lucky_win"
            elif outcome == "TIE":
                trade_type = "neutral_tie"
            else:
                trade_type = "unknown"
        d["trade_type"] = trade_type

        # Mantener loss_type por compatibilidad
        if outcome == "LOSS":
            d["loss_type"] = trade_type

        if "failed_blocks" not in d or not isinstance(d["failed_blocks"], list):
            fb = []
            for k in ("context", "ema21", "ema3_6", "entry_candle", "no_entry_filters"):
                if str(checklist.get(k, "unclear")).lower() == "fail":
                    fb.append(k)
            d["failed_blocks"] = fb

        fixed.append(d)

    out["trade_reviews"] = fixed
    # Mantener loss_trade_reviews por compatibilidad
    out["loss_trade_reviews"] = [t for t in fixed if t.get("outcome") == "LOSS"]

    # Anti-contradicción para resumen
    any_invalid = any((it.get("ai_validity") is False) for it in fixed)
    if any_invalid:
        summ = str(out.get("session_summary_md", ""))
        if "todas" in summ.lower() and "checklist" in summ.lower():
            out["session_summary_md"] = summ + "\n\n⚠️ Nota: Hubo violaciones del protocolo A+ (ai_validity=false en al menos un LOSS)."

    return out


def analyze_session_with_vision(
    *,
    session_payload: Dict[str, Any],
    max_output_tokens: int = 1400,
) -> Dict[str, Any]:
    client = _client()
    model = _model()

    scope = str((session_payload.get("session_meta") or {}).get("scope", "session")).lower()
    prompt = PROMPT_REVIEW_WEEKLY_V1 if scope == "weekly" else PROMPT_REVIEW_SESSION_V1
    prompt_version = "review_weekly_v1.0" if scope == "weekly" else "review_session_v1.0"

    payload_text = _json_dumps_safe(session_payload)
    payload_hash = _hash_payload(payload_text)

    # Estimación SOLO texto (las imágenes no entran)
    est_prompt_tokens = _count_text_tokens(model, prompt + payload_text)

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": "Datos de la sesión (JSON):"},
        {"type": "input_text", "text": payload_text},
    ]

    trades_imgs = session_payload.get("trades_with_images") or session_payload.get("losses_with_images") or []
    for i, t in enumerate(trades_imgs, start=1):
        url = _safe_str(t.get("screenshot_url")).strip()
        outcome = str(t.get("outcome", "UNKNOWN")).upper()
        if url:
            content.append(
                {
                    "type": "input_text",
                    "text": f"Imagen #{i} (outcome={outcome}) trade_id={t.get('id')} (analiza con checklist A+ y reglas duras):",
                }
            )
            content.append({"type": "input_image", "image_url": url})

    started_at = datetime.now(timezone.utc).isoformat()

    parse_retries = 0
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
    )

    def _try_parse(r) -> Dict[str, Any]:
        t = getattr(r, "output_text", None) or str(r)
        return _extract_json(t)

    def _is_valid_review(d: Dict[str, Any]) -> bool:
        if not isinstance(d, dict):
            return False
        if not d.get("session_summary_md"):
            return False
        cf = d.get("checklist_findings")
        if not isinstance(cf, dict):
            return False
        required_cf = ["pass_vs_fail_interpretation", "most_common_breaks", "win_analysis", "loss_analysis", "tie_analysis"]
        if any(k not in cf for k in required_cf):
            return False
        rules = d.get("action_rules")
        if not isinstance(rules, list) or len(rules) == 0:
            return False
        return True

    data: Dict[str, Any]
    try:
        data = _try_parse(resp)
        data = normalize_review(data)
        if not _is_valid_review(data):
            raise ValueError("review_incomplete")
    except Exception:
        # Reintento con instrucción explícita de JSON estricto y campos obligatorios
        parse_retries = 1
        retry_content = content + [
            {
                "type": "input_text",
                "text": (
                    "IMPORTANTE: Responde SOLO con JSON válido, sin texto adicional. "
                    "No uses comillas simples ni comentarios. Sin trailing commas. "
                    "Incluye SIEMPRE session_summary_md, checklist_findings completo "
                    "y action_rules (mínimo 3)."
                ),
            }
        ]
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": retry_content}],
            max_output_tokens=max_output_tokens,
        )
        data = _try_parse(resp)
        data = normalize_review(data)

    ended_at = datetime.now(timezone.utc).isoformat()

    usage = _usage_to_dict(resp)
    prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    # Meta (para UI/log/caching)
    data["_meta"] = {
        "model": model,
        "prompt_version": prompt_version,
        "payload_hash": payload_hash,
        "started_at": started_at,
        "ended_at": ended_at,
        "estimated_text_prompt_tokens": est_prompt_tokens,
        "parse_retries": parse_retries,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    return data
