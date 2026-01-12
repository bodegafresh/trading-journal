from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

ChecklistVerdict = Literal["pass", "fail", "unclear"]

# -----------------------------
# 1) Checklist "machine-readable"
# -----------------------------
A_PLUS_CHECKLIST: Dict[str, Any] = {
    "name": "A+ Entry Checklist v1.0",
    "version": "1.0",
    "blocks": [
        {
            "key": "context",
            "title": "Contexto (estructura)",
            "rules": [
                "Mercado NO lateral",
                "Bollinger expandiéndose (no comprimida)",
                "Dirección clara en últimos impulsos",
            ],
        },
        {
            "key": "ema21",
            "title": "Tendencia micro (EMA21)",
            "rules": [
                "Precio arriba EMA21 -> solo CALL (UP)",
                "Precio abajo EMA21 -> solo PUT (DOWN)",
                "EMA21 con pendiente clara (no plana)",
            ],
        },
        {
            "key": "ema3_6",
            "title": "Momentum (EMA3/EMA6)",
            "rules": [
                "EMA3 y EMA6 alineadas con la dirección",
                "No cruces caóticos recientes",
                "Pullback ordenado o continuación limpia",
            ],
        },
        {
            "key": "entry_candle",
            "title": "Vela de entrada (timing)",
            "rules": [
                "Cuerpo claro",
                "Sin mechas largas en contra",
                "No es entrada tardía (no perseguir vela)",
            ],
        },
        {
            "key": "no_entry_filters",
            "title": "Filtros NO entrada (estado interno)",
            "rules": [
                "No aburrimiento",
                "No urgencia",
                "No vengo de 2 pérdidas seguidas",
                "No fuera de horario definido",
            ],
        },
    ],
}

BLOCK_KEYS = [b["key"] for b in A_PLUS_CHECKLIST["blocks"]]


def checklist_as_prompt_text() -> str:
    """Devuelve texto compacto para meter en prompts de IA."""
    lines = [f"{A_PLUS_CHECKLIST['name']} (v{A_PLUS_CHECKLIST['version']})", ""]
    for b in A_PLUS_CHECKLIST["blocks"]:
        lines.append(f"- {b['title']}:")
        for r in b["rules"]:
            lines.append(f"  • {r}")
    return "\n".join(lines)


# -----------------------------
# 2) Score A+ automático por trade
# -----------------------------
@dataclass(frozen=True)
class APlusScore:
    score_0_5: float
    score_0_1: float
    is_a_plus: bool
    details: Dict[str, ChecklistVerdict]


def _verdict_to_points(v: ChecklistVerdict) -> float:
    # pass=1, unclear=0.5 (evidencia insuficiente), fail=0
    if v == "pass":
        return 1.0
    if v == "unclear":
        return 0.5
    return 0.0


def score_from_block_verdicts(blocks: Dict[str, ChecklistVerdict]) -> APlusScore:
    """
    blocks: {"context": "pass/fail/unclear", ...}
    Devuelve score y bandera A+ (solo si 5/5 en pass).
    """
    details: Dict[str, ChecklistVerdict] = {}
    total = 0.0

    for k in BLOCK_KEYS:
        v = blocks.get(k, "unclear")
        details[k] = v
        total += _verdict_to_points(v)

    score_0_5 = round(total, 2)
    score_0_1 = round(score_0_5 / 5.0, 3)
    is_a_plus = all(details[k] == "pass" for k in BLOCK_KEYS)

    return APlusScore(
        score_0_5=score_0_5,
        score_0_1=score_0_1,
        is_a_plus=is_a_plus,
        details=details,
    )


def score_from_db_checklist_flag(checklist_pass: Optional[bool]) -> APlusScore:
    """
    Si solo tienes el booleano en BD:
      - True  => 5/5
      - False => 0/5
      - None  => 2.5/5 (unknown)
    """
    if checklist_pass is True:
        blocks = {k: "pass" for k in BLOCK_KEYS}
        return score_from_block_verdicts(blocks)
    if checklist_pass is False:
        blocks = {k: "fail" for k in BLOCK_KEYS}
        return score_from_block_verdicts(blocks)

    blocks = {k: "unclear" for k in BLOCK_KEYS}
    return score_from_block_verdicts(blocks)


def combine_scores(
    *,
    db_flag: Optional[bool],
    vision_blocks: Optional[Dict[str, ChecklistVerdict]] = None,
) -> APlusScore:
    """
    Preferencia:
      - si hay visión (loss con screenshot) => score por bloques IA
      - si no => fallback al flag checklist_pass
    """
    if vision_blocks:
        return score_from_block_verdicts(vision_blocks)
    return score_from_db_checklist_flag(db_flag)
