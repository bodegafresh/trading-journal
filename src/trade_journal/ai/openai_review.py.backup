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
# Prompt v1.0 definitivo
# ----------------------------
PROMPT_REVIEW_V1 = r"""
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
- loss_type:
  - Si outcome == LOSS y ai_validity == true => good_loss
  - Si outcome == LOSS y ai_validity == false => bad_loss
  - Si no => neutral
- Prohibido afirmar “pasó checklist” si ai_validity=false.
- Nada de consejos genéricos: cada corrección debe ser una acción concreta aplicable en el próximo trade.

Input:
Recibirás JSON con:
- session_meta
- kpis
- trades_sample
- losses_with_images (cada item incluye screenshot_url público)

Tu tarea:
1) Entregar resumen de sesión SIN contradicciones, basado en datos.
2) Auditar cada LOSS con evidencia (imagen) y llenar checklist por bloques.
3) Para cada LOSS: listar failed_blocks y failed_rules (reglas específicas dentro de cada bloque).
4) Producir reglas accionables de la sesión (máx 5, concretas).

FORMATO DE SALIDA:
Devuelve SOLO JSON válido con esta estructura:

{
  "session_summary_md": "markdown breve (máx ~20 líneas)",
  "checklist_findings": {
    "pass_vs_fail_interpretation": "texto corto",
    "most_common_breaks": ["...", "..."]
  },
  "action_rules": ["...", "...", "...", "...", "..."],
  "loss_trade_reviews": [
    {
      "trade_id": "uuid",
      "ai_score_a_plus_0_5": 0,
      "ai_validity": true,
      "ai_is_a_plus": false,
      "loss_type": "good_loss|bad_loss|neutral",
      "checklist": {
        "context": "pass|fail|unclear",
        "ema21": "pass|fail|unclear",
        "ema3_6": "pass|fail|unclear",
        "entry_candle": "pass|fail|unclear",
        "no_entry_filters": "pass|fail|unclear"
      },
      "failed_blocks": ["context", "entry_candle"],
      "failed_rules": ["Bollinger comprimida", "Entrada tardía (persecución)"],
      "primary_cause": "frase corta",
      "one_fix": "1 acción concreta para el próximo trade",
      "confidence_0_1": 0.0
    }
  ]
}
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
    max_losses_with_images: int = 10,
) -> Dict[str, Any]:
    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]
    ties = [t for t in trades if t.outcome == "TIE"]

    losses_with_img = [
        t for t in losses
        if t.screenshot_url and _is_image_url(t.screenshot_url)
    ][:max_losses_with_images]

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
            "losses_with_images_used": len(losses_with_img),
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
        "losses_with_images": [
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
            for t in losses_with_img
        ],
    }


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _enforce_hard_rules(review: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(review or {})
    ltr = out.get("loss_trade_reviews") or []
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

        loss_type = d.get("loss_type")
        if not loss_type:
            loss_type = "good_loss" if ai_validity else "bad_loss"
        d["loss_type"] = loss_type

        if "failed_blocks" not in d or not isinstance(d["failed_blocks"], list):
            fb = []
            for k in ("context", "ema21", "ema3_6", "entry_candle", "no_entry_filters"):
                if str(checklist.get(k, "unclear")).lower() == "fail":
                    fb.append(k)
            d["failed_blocks"] = fb

        fixed.append(d)

    out["loss_trade_reviews"] = fixed

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

    payload_text = _json_dumps_safe(session_payload)
    payload_hash = _hash_payload(payload_text)

    # Estimación SOLO texto (las imágenes no entran)
    est_prompt_tokens = _count_text_tokens(model, PROMPT_REVIEW_V1 + payload_text)

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": PROMPT_REVIEW_V1},
        {"type": "input_text", "text": "Datos de la sesión (JSON):"},
        {"type": "input_text", "text": payload_text},
    ]

    losses_imgs = session_payload.get("losses_with_images") or []
    for i, t in enumerate(losses_imgs, start=1):
        url = _safe_str(t.get("screenshot_url")).strip()
        if url:
            content.append(
                {
                    "type": "input_text",
                    "text": f"Imagen LOSS #{i} trade_id={t.get('id')} (analiza con checklist A+ y reglas duras):",
                }
            )
            content.append({"type": "input_image", "image_url": url})

    started_at = datetime.now(timezone.utc).isoformat()

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
    )

    ended_at = datetime.now(timezone.utc).isoformat()

    usage = _usage_to_dict(resp)
    prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    text = getattr(resp, "output_text", None) or str(resp)
    data = _extract_json(text)
    data = _enforce_hard_rules(data)

    # Meta (para UI/log/caching)
    data["_meta"] = {
        "model": model,
        "prompt_version": "review_v1.0",
        "payload_hash": payload_hash,
        "started_at": started_at,
        "ended_at": ended_at,
        "estimated_text_prompt_tokens": est_prompt_tokens,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    return data
