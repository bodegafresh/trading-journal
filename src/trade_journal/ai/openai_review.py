from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from trade_journal.ai.checklist_a_plus import BLOCK_KEYS, checklist_as_prompt_text

# Import perezoso (evita pantallazo rojo si falta openai en el venv)
def _client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar 'openai'.\n"
            "Si usas Poetry: ejecuta `poetry add openai` y corre con `poetry run streamlit run ...`.\n"
            "Si usas pip: instala en el MISMO intérprete que corre Streamlit.\n"
            f"Error original: {e}"
        )
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
    return u.endswith((".png", ".jpg", ".jpeg", ".webp"))


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

    total = len(trades)
    wr_with_ties = (len(wins) / total) if total else 0.0
    decided = len(wins) + len(losses)
    wr_no_ties = (len(wins) / decided) if decided else 0.0
    tie_rate = (len(ties) / total) if total else 0.0
    pnl_total = sum(t.pnl for t in trades)
    ev_r = (sum(t.r_mult for t in trades) / total) if total else 0.0

    chk_pass = sum(1 for t in trades if t.checklist_pass is True)
    chk_fail = sum(1 for t in trades if t.checklist_pass is False)
    chk_none = sum(1 for t in trades if t.checklist_pass is None)

    return {
        "session_meta": session_meta,
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
            "checklist": {"pass": chk_pass, "fail": chk_fail, "missing": chk_none},
            "losses_with_images_used": len(losses_with_img),
        },
        "trades_sample": [
            {
                "id": t.id,
                "trade_time": t.trade_time,
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
                "trade_time": t.trade_time,
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


def analyze_session_with_vision(
    *,
    session_payload: Dict[str, Any],
    max_output_tokens: int = 1600,
) -> Dict[str, Any]:
    client = _client()
    model = _model()

    system = (
        "Eres un auditor cuantitativo y de ejecución. "
        "No das señales ni propones indicadores nuevos. "
        "Evalúas disciplina A+ (binaria), sesgos y causas repetibles. "
        "Sé directo y operativo. Si falta evidencia, responde 'unclear'."
    )

    checklist_text = checklist_as_prompt_text()

    # Contrato de salida: JSON estricto (sin texto extra)
    output_contract = {
        "session_summary_md": "markdown breve",
        "execution_quality": {
            "a_plus_adherence_score_0_1": 0.0,
            "primary_failure_modes": ["..."],
        },
        "loss_trade_reviews": [
            {
                "trade_id": "uuid",
                "a_plus_score_0_5": 0.0,
                "blocks": {k: "pass|fail|unclear" for k in BLOCK_KEYS},
                "primary_cause": "frase corta",
                "correction": "acción concreta y testeable",
                "confidence_0_1": 0.0,
            }
        ],
    }

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": system},
        {"type": "input_text", "text": "Checklist oficial (inmutable):\n" + checklist_text},
        {"type": "input_text", "text": "Datos de la sesión (JSON):\n" + json.dumps(session_payload, ensure_ascii=False)},
        {"type": "input_text", "text": "Devuelve SOLO JSON válido siguiendo este contrato:\n" + json.dumps(output_contract, ensure_ascii=False)},
    ]

    losses_imgs = session_payload.get("losses_with_images") or []
    for i, t in enumerate(losses_imgs, start=1):
        url = str(t.get("screenshot_url") or "").strip()
        if url:
            content.append({"type": "input_text", "text": f"Imagen LOSS #{i} trade_id={t.get('id')} — evalúa bloques A+."})
            content.append({"type": "input_image", "image_url": url})

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
    )

    text = getattr(resp, "output_text", "") or ""
    if not text:
        raise RuntimeError("OpenAI no devolvió output_text.")

    # Parse JSON robusto
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise RuntimeError(f"No se pudo parsear JSON. Respuesta:\n{text}")
