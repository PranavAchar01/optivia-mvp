"""
Reflection sub-steps (§5.3.4, §5.3.5, §5.3.10).

A lightweight Haiku call that scores an agent's output against a per-agent
rubric and the retrieved lessons L_t. Returns (critique, confidence ∈ [0,1]).
The calling agent re-runs itself with the critique if confidence ≤ 0.7.

Selective-skip: if the moving-average quality is high AND it isn't the first
turn AND the output text overlaps the retrieved lessons, skip the call and
return (output, 1.0). This is the "minimise latency when stable" branch.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings

log = structlog.get_logger(__name__)

client = instructor.from_anthropic(AsyncAnthropic(api_key=settings.anthropic_api_key))

CONFIDENCE_THRESHOLD = 0.7
SKIP_QUALITY_THRESHOLD = 0.85


class ReflectionVerdict(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    critique: str = ""
    consistent_with_lessons: bool = True


_REFLECT_SYSTEM = """\
You are Optivia's Reflection Step. Given an agent's output and a rubric,
return:
  • confidence ∈ [0,1] — how well the output satisfies the rubric AND aligns
    with the retrieved lessons.
  • critique — one sentence describing what to fix if confidence ≤ 0.7.
Be terse. Never paraphrase the output. If the output contradicts a lesson,
say so explicitly and set consistent_with_lessons=false.
"""


def _diverges_from_lessons(output: str, lessons: list[Any]) -> bool:
    if not lessons:
        return False
    out = output.lower()
    out_tokens = set(re.findall(r"[a-z0-9_]{4,}", out))
    for l in lessons:
        text = getattr(l, "lesson", None) or (l if isinstance(l, str) else "")
        if not text:
            continue
        text_tokens = set(re.findall(r"[a-z0-9_]{4,}", text.lower()))
        if not text_tokens:
            continue
        # Heuristic: shared topic + opposing modal
        if (
            len(out_tokens & text_tokens) >= 2
            and ("never" in text.lower() and "must" in out)
        ):
            return True
    return False


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4), reraise=True)
async def _call_reflect(prompt: str) -> ReflectionVerdict:
    return await client.chat.completions.create(
        model=settings.model_haiku,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
        system=_REFLECT_SYSTEM,
        response_model=ReflectionVerdict,
    )


async def reflect(
    *,
    agent_name: str,
    output: str,
    rubric: str,
    lessons: list[Any] | None = None,
    input_context: str = "",
    avg_quality: float = 0.0,
    is_first_turn: bool = True,
) -> tuple[str, float]:
    """
    Returns (critique, confidence). The caller decides whether to re-run.

    Selective-skip: stable session AND aligned with lessons → skip the call.
    """
    lessons = lessons or []

    if (
        avg_quality >= SKIP_QUALITY_THRESHOLD
        and not is_first_turn
        and not _diverges_from_lessons(output, lessons)
    ):
        log.info("reflect.skip", agent=agent_name, reason="stable_aligned")
        return "", 1.0

    lesson_block = ""
    for i, l in enumerate(lessons[:5]):
        text = getattr(l, "lesson", None) or (l if isinstance(l, str) else "")
        if text:
            lesson_block += f"  [{i+1}] {text}\n"

    prompt = (
        f"Agent under review: {agent_name}\n"
        f"Rubric: {rubric}\n"
        f"Retrieved lessons:\n{lesson_block or '  (none)'}\n"
        f"Input context: {input_context[:600]}\n\n"
        f"Agent output:\n\"\"\"\n{output[:1200]}\n\"\"\""
    )

    try:
        verdict = await _call_reflect(prompt)
        return verdict.critique, float(verdict.confidence)
    except Exception as exc:
        log.warning("reflect.error", agent=agent_name, error=str(exc))
        # Fail open at neutral confidence so the caller doesn't loop forever
        return "", 0.75
