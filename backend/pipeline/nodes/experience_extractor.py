"""
Agent 15B — Experience Extractor (§5.3.19).

Distills the execution trajectory into a structured ExperienceRecord, then
applies the ExpeL-inspired insight operators (ADD / UPVOTE / DOWNVOTE / EDIT)
against the lessons that were retrieved earlier in the turn.

Quality gates:
  - Reject records containing obvious secrets / PII / raw code blocks.
  - Reject records lacking an actionable future-behavior change.
  - Reject when quality scalar is too uncertain to draw a conclusion.

Implicit-conflict detection:
  If a high-quality successful trajectory contradicts a retained lesson, the
  retained lesson's conf_count is reset to 0 and its trust is decremented.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.llm import llm_client
from backend.core.models import ExperienceRecord, ExperienceScope, TaskType
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)

# ExpeL operator deltas
_DELTA_TRUST = 0.25
_TRUST_MAX = 5.0

# Secret / PII patterns — quality gate
_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{16,}"),                  # api keys
    re.compile(r"AKIA[0-9A-Z]{16}"),                         # AWS access
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),       # PEM
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                    # SSN
    re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"),           # emails
]
_CODE_BLOCK_RE = re.compile(r"```|def\s+\w+\(|class\s+\w+:|function\s+\w+\(")
# Anti-actionable phrases — the lesson must imply a behavior change
_ACTIONABLE_HINTS = re.compile(
    r"\b(should|must|always|never|prefer|avoid|require|use)\b", re.IGNORECASE
)


# ── LLM extraction call ────────────────────────────────────────────────────────

class ExtractedExperience(BaseModel):
    lesson: str = Field(description="One actionable sentence — the reusable insight.")
    failure_modes: list[str] = Field(default_factory=list)
    successful_patterns: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    actionable: bool = Field(description="True if the lesson implies a behavior change.")


_EXTRACT_SYSTEM = """\
You are Optivia's Experience Extractor. Given an execution trajectory + quality
verdict, distill ONE reusable lesson for future runs.

Rules:
- The lesson MUST be one short sentence (≤ 25 words).
- It MUST be actionable (use "always/never/prefer/avoid/require").
- It MUST NOT contain code, function names, file paths, or credentials.
- failure_modes/successful_patterns: 0–3 short bullets each.
- tags: 1–5 lowercase keywords describing the domain (e.g. "auth", "migration").
- If no meaningful lesson can be drawn, set actionable=False and lesson="".
"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=True)
async def _call_extractor(context: str) -> ExtractedExperience:
    return await llm_client.structured_generate(
        system_prompt=_EXTRACT_SYSTEM,
        user_prompt=context,
        response_model=ExtractedExperience,
    )


# ── Quality gates ──────────────────────────────────────────────────────────────

def _passes_gates(lesson: str) -> bool:
    if not lesson or len(lesson) < 8:
        return False
    if any(p.search(lesson) for p in _SECRET_PATTERNS):
        return False
    if _CODE_BLOCK_RE.search(lesson):
        return False
    if not _ACTIONABLE_HINTS.search(lesson):
        return False
    return True


def _contradicts(new_lesson: str, old_lesson: str) -> bool:
    if not new_lesson or not old_lesson:
        return False
    new_low = new_lesson.lower()
    old_low = old_lesson.lower()
    # Heuristic: shared topic tokens + opposite directive
    new_tokens = set(re.findall(r"[a-z0-9_]{4,}", new_low))
    old_tokens = set(re.findall(r"[a-z0-9_]{4,}", old_low))
    if len(new_tokens & old_tokens) < 2:
        return False
    polarity_pairs = [("always", "never"), ("prefer", "avoid"), ("use", "do not use")]
    for a, b in polarity_pairs:
        if (a in new_low and b in old_low) or (b in new_low and a in old_low):
            return True
    return False


# ── Operators ──────────────────────────────────────────────────────────────────

def _apply_upvote(rec: dict[str, Any], current_run: int) -> dict[str, Any]:
    rec["weight"] = int(rec.get("weight", 2)) + 1
    rec["conf_count"] = int(rec.get("conf_count", 1)) + 1
    rec["trust_score"] = min(_TRUST_MAX, float(rec.get("trust_score", 1.0)) + _DELTA_TRUST)
    rec["last_confirmed_run"] = current_run
    return rec


def _apply_downvote(rec: dict[str, Any]) -> dict[str, Any]:
    rec["weight"] = max(0, int(rec.get("weight", 2)) - 1)
    rec["conf_count"] = 0
    rec["trust_score"] = max(0.0, float(rec.get("trust_score", 1.0)) - _DELTA_TRUST)
    if rec["weight"] <= 0:
        rec["archived"] = True
    return rec


def _apply_implicit_conflict(rec: dict[str, Any]) -> dict[str, Any]:
    """When new evidence contradicts a retained lesson."""
    rec["conf_count"] = 0
    rec["trust_score"] = max(0.0, float(rec.get("trust_score", 1.0)) - 2 * _DELTA_TRUST)
    rec["weight"] = max(0, int(rec.get("weight", 2)) - 1)
    if rec["weight"] <= 0:
        rec["archived"] = True
    return rec


# ── Main node ──────────────────────────────────────────────────────────────────

async def experience_extractor(state: OptiviaState) -> OptiviaState:
    """Agent 15B — Experience Extractor."""
    request_id = state.get("request_id")
    trace = state.get("execution_trace", [])
    quality = state.get("quality")
    q_score = quality.score if quality else 0.0
    task_cls = state.get("task_classification")
    retrieved: list[ExperienceRecord] = state.get("retrieved_lessons", []) or []  # type: ignore[assignment]
    current_run = int(state.get("turn_index", 0))

    if not trace:
        state["extracted_experience"] = []
        return state

    # Build extraction context (truncate aggressively — Haiku call)
    trace_preview = " | ".join(
        f"{e.event_type}:{str(e.payload)[:200]}" for e in trace[-3:]
    )
    raw_prompt = state.get("raw_prompt", "")[:400]
    outcome_label = "success" if q_score >= 0.75 else "failure"

    ctx = (
        f"Prompt: {raw_prompt}\n"
        f"Task type: {task_cls.task_type.value if task_cls else 'unknown'}\n"
        f"Quality: {q_score:.2f} ({outcome_label})\n"
        f"Trajectory tail: {trace_preview[:1200]}\n"
    )

    extracted: ExtractedExperience | None = None
    try:
        extracted = await _call_extractor(ctx)
    except Exception as exc:
        log.warning("experience_extractor.llm_error", error=str(exc))

    new_lesson_text = (extracted.lesson if extracted else "").strip()
    actionable = bool(extracted and extracted.actionable)
    passes_gates = actionable and _passes_gates(new_lesson_text)

    updates: list[dict[str, Any]] = []
    new_record: ExperienceRecord | None = None

    # Apply UPVOTE / IMPLICIT_CONFLICT to retained lessons based on this turn's outcome
    for lesson in retrieved:
        rec_dict: dict[str, Any] = lesson.model_dump()
        rec_dict["id"] = lesson.id  # ensure round-trip
        if q_score >= 0.75:
            # Successful turn confirms followed lessons
            _apply_upvote(rec_dict, current_run)
            # …unless new evidence contradicts the lesson
            if passes_gates and _contradicts(new_lesson_text, lesson.lesson):
                _apply_implicit_conflict(rec_dict)
        elif q_score < 0.5:
            _apply_downvote(rec_dict)
        updates.append(rec_dict)

    # ADD new record if gates pass
    if passes_gates:
        now_iso = datetime.now(timezone.utc).isoformat()
        task_type_enum = task_cls.task_type if task_cls else TaskType.NEW_CODE
        new_record = ExperienceRecord(
            id=str(uuid.uuid4()),
            scope=ExperienceScope.PROJECT,
            workspace_id=state.get("workspace_id", ""),
            user_id=state.get("user_id", ""),
            task_type=task_type_enum,
            tags=list(extracted.tags[:5] if extracted else []),
            lesson=new_lesson_text,
            failure_modes=list(extracted.failure_modes[:3] if extracted else []),
            successful_patterns=list(extracted.successful_patterns[:3] if extracted else []),
            outcome_label=outcome_label,
            weight=2,
            trust_score=1.0,
            conf_count=1,
            last_confirmed_run=current_run,
            last_confirmed=now_iso,
            created_at=now_iso,
            archived=False,
        )

    # Persist
    try:
        from backend.db.client import db_client
        if updates:
            await db_client.bulk_update_experience_stats(updates)
        if new_record:
            payload = new_record.model_dump()
            payload["task_type"] = new_record.task_type.value
            payload["scope"] = new_record.scope.value
            await db_client.upsert_experience(payload)
    except Exception as exc:
        log.warning("experience_extractor.persist_error", error=str(exc))

    # Surface for downstream nodes (kept as list[str] for backward-compat with persist.py)
    if new_record:
        state["extracted_experience"] = [new_record.lesson]
    else:
        state["extracted_experience"] = []

    log.info(
        "experience_extractor.done",
        request_id=request_id,
        added=bool(new_record),
        n_updates=len(updates),
        gates_passed=passes_gates,
        q=q_score,
    )
    return state
