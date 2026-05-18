"""
Agent 2B — Experience Retriever (§5.3.3).

Implements:
  • Retrieval scoring  s(e,p,τ,T) = w_τ·𝟙(τ_e=τ) + w_T·|T∩T_e| + w_p·sim(p,l_e) + w_o·B(o_e)
  • Cross-scope conflict resolution via
        effective_priority(e) = scope_weight · trust · recency_factor
  • Hard security boundary: global ↛ project (the global lesson is dropped
    when it textually contradicts any retained project-scoped lesson).
  • Staleness filter: archive when
        (current_run - last_confirmed_run) > N_stale   OR
        days_since(last_confirmed)        > D_stale
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

import structlog

from backend.core.models import ExperienceRecord, ExperienceScope
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)

# Retrieval scoring weights
_W_TAU = 0.30
_W_TAG = 0.20
_W_SIM = 0.35
_W_OUT = 0.15

# Outcome bonus B(o_e)
_B_SUCCESS = 0.5
_B_FAILURE = 0.25

# Cross-scope priority weights (§5.3.3)
_SCOPE_WEIGHTS = {"project": 3.0, "user": 2.0, "global": 1.0}

# Recency decay
_LAMBDA_REC = 0.02   # per day

# Staleness thresholds
_N_STALE = 20
_D_STALE = 30

_TOP_K = 5


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]{3,}", (text or "").lower())}


def _keyword_sim(a: str, b: str) -> float:
    """Lightweight cosine over token sets (MVP — no embedding round-trip)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / math.sqrt(len(ta) * len(tb))


def _outcome_bonus(label: str) -> float:
    return _B_SUCCESS if label == "success" else _B_FAILURE


def _recency_factor(last_confirmed: Any) -> float:
    if not last_confirmed:
        return 0.5
    try:
        if isinstance(last_confirmed, str):
            ts = datetime.fromisoformat(last_confirmed.replace("Z", "+00:00"))
        else:
            ts = last_confirmed
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        days = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
        return math.exp(-_LAMBDA_REC * days)
    except Exception:
        return 0.5


def _is_stale(rec: dict[str, Any], current_run: int) -> bool:
    last_run = int(rec.get("last_confirmed_run", 0))
    if (current_run - last_run) > _N_STALE:
        return True
    factor = _recency_factor(rec.get("last_confirmed"))
    # exp(-0.02 * D_STALE) ≈ 0.549 — below this means past D_STALE days
    return factor < math.exp(-_LAMBDA_REC * _D_STALE)


def _retrieval_score(
    rec: dict[str, Any],
    prompt: str,
    task_type: str,
    prompt_tags: set[str],
) -> float:
    tau_match = 1.0 if rec.get("task_type") == task_type else 0.0
    tags = set(rec.get("tags") or [])
    tag_overlap = len(tags & prompt_tags) / max(1, len(prompt_tags))
    sim = _keyword_sim(prompt, rec.get("lesson", ""))
    out_b = _outcome_bonus(rec.get("outcome_label", "success"))
    return _W_TAU * tau_match + _W_TAG * tag_overlap + _W_SIM * sim + _W_OUT * out_b


def _effective_priority(rec: dict[str, Any]) -> float:
    scope = rec.get("scope", "project")
    return (
        _SCOPE_WEIGHTS.get(scope, 1.0)
        * float(rec.get("trust_score", 1.0))
        * _recency_factor(rec.get("last_confirmed"))
    )


def _resolve_cross_scope_conflicts(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (kept, downvoted). Two records textually contradict each other
    when their token sets overlap heavily but lessons differ. Highest
    effective_priority wins; the loser is downvoted.

    Hard security rule: a global-scope record can NEVER override a
    project-scope record — even if its computed priority is higher.
    """
    kept: list[dict[str, Any]] = []
    downvoted: list[dict[str, Any]] = []

    def contradicts(a: dict[str, Any], b: dict[str, Any]) -> bool:
        if a.get("lesson") == b.get("lesson"):
            return False
        sim = _keyword_sim(a.get("lesson", ""), b.get("lesson", ""))
        return sim > 0.6

    sorted_records = sorted(records, key=_effective_priority, reverse=True)
    for rec in sorted_records:
        conflict = next((k for k in kept if contradicts(rec, k)), None)
        if conflict is None:
            kept.append(rec)
            continue
        # Security boundary
        if rec.get("scope") == "project" and conflict.get("scope") == "global":
            downvoted.append(conflict)
            kept = [k for k in kept if k is not conflict]
            kept.append(rec)
        elif rec.get("scope") == "global" and conflict.get("scope") == "project":
            downvoted.append(rec)  # global dropped outright
        else:
            downvoted.append(rec)

    for d in downvoted:
        d["conf_count"] = 0
        d["trust_score"] = max(0.0, float(d.get("trust_score", 1.0)) - 0.5)

    return kept, downvoted


async def experience_retriever(state: OptiviaState) -> OptiviaState:
    """Agent 2B — Experience Retriever."""
    request_id = state.get("request_id")
    raw = state.get("raw_prompt", "")
    task_cls = state.get("task_classification")
    task_type = task_cls.task_type.value if task_cls else None
    workspace_id = state.get("workspace_id", "")
    user_id = state.get("user_id", "")
    current_run = int(state.get("turn_index", 0))

    candidates: list[dict[str, Any]] = []
    try:
        from backend.db.client import db_client
        candidates = await db_client.get_experiences_for_retrieval(
            workspace_id=workspace_id,
            user_id=user_id,
            task_type=task_type,
            limit=50,
        )
    except Exception as exc:
        log.warning("experience_retriever.db_error", error=str(exc))
        candidates = []

    # Staleness filter (archive in-place, don't return)
    stale: list[dict[str, Any]] = []
    fresh: list[dict[str, Any]] = []
    for r in candidates:
        if _is_stale(r, current_run):
            r["archived"] = True
            stale.append(r)
        else:
            fresh.append(r)

    prompt_tags = _tokens(raw)
    for r in fresh:
        r["_score"] = _retrieval_score(r, raw, task_type or "", prompt_tags)

    # Rank by retrieval score, then prune by cross-scope conflict resolution
    fresh.sort(key=lambda r: r["_score"], reverse=True)
    top = fresh[: max(_TOP_K * 2, _TOP_K)]
    kept, downvoted = _resolve_cross_scope_conflicts(top)
    kept = kept[:_TOP_K]

    # Materialise into ExperienceRecord models for downstream agents
    lessons: list[ExperienceRecord] = []
    for r in kept:
        try:
            lessons.append(
                ExperienceRecord(
                    id=str(r.get("id", "")),
                    scope=ExperienceScope(r.get("scope", "project")),
                    workspace_id=str(r.get("workspace_id", "")),
                    user_id=str(r.get("user_id", "")),
                    task_type=r.get("task_type", "new_code"),  # type: ignore[arg-type]
                    tags=list(r.get("tags") or []),
                    lesson=r.get("lesson", ""),
                    failure_modes=list(r.get("failure_modes") or []),
                    successful_patterns=list(r.get("successful_patterns") or []),
                    outcome_label=r.get("outcome_label", "success"),  # type: ignore[arg-type]
                    weight=int(r.get("weight", 2)),
                    trust_score=float(r.get("trust_score", 1.0)),
                    conf_count=int(r.get("conf_count", 1)),
                    last_confirmed_run=int(r.get("last_confirmed_run", 0)),
                )
            )
        except Exception:
            continue

    state["retrieved_lessons"] = lessons
    state["_retrieved_lesson_ids"] = [l.id for l in lessons]  # type: ignore[typeddict-unknown-key]

    # Persist staleness + downvote updates
    try:
        from backend.db.client import db_client
        if stale:
            await db_client.bulk_update_experience_stats(stale)
        if downvoted:
            await db_client.bulk_update_experience_stats(downvoted)
    except Exception as exc:
        log.warning("experience_retriever.update_error", error=str(exc))

    # Token accounting — every lesson string costs roughly len/4 tokens
    mem_tokens = sum(len(l.lesson) for l in lessons) // 4
    state["memory_tokens"] = state.get("memory_tokens", 0) + mem_tokens

    log.info(
        "experience_retriever.done",
        request_id=request_id,
        n_candidates=len(candidates),
        n_fresh=len(fresh),
        n_kept=len(lessons),
        n_stale_archived=len(stale),
        n_downvoted=len(downvoted),
    )
    return state
