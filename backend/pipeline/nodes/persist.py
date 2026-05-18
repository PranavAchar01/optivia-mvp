"""
Agent 17 — Session Persister (§5, §6.4, §10.1).

Responsibilities:
- Write the full OptiviaState to Postgres traces table (Trace Contract)
- Upsert the SessionState row for multi-turn continuity
- Update Redis semantic cache if execution was user-accepted
- Emit Langfuse score telemetry (§9.1)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

import structlog

from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _j(obj: Any) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return {"items": [_j(i) for i in obj]}
    if isinstance(obj, dict):
        return obj
    return {}


def _prompt_hash_bytes(raw: str) -> bytes:
    return hashlib.sha256(raw.encode()).digest()


# ── Main node ─────────────────────────────────────────────────────────────────

async def session_persister(state: OptiviaState) -> OptiviaState:
    """Agent 17 — Session Persister."""
    trace_id = state.get("trace_id") or str(uuid.uuid4())
    state["trace_id"] = trace_id

    scores = state.get("scores_updated") or state.get("scores")
    routing = state.get("routing_decision")
    master = state.get("master_prompt")
    outcome = state.get("outcome")
    quality = state.get("quality")

    # ── 1. Write trace row ────────────────────────────────────────────────────
    # Compute the embedding if session_loader didn't (cache-hit path skips it).
    embedding = state.get("_prompt_embedding")  # type: ignore[typeddict-item]
    if embedding is None:
        try:
            from backend.embeddings import embed
            embedding = await embed(state.get("raw_prompt", ""), input_type="document")
        except Exception:
            embedding = None

    trace_record = {
        "id": trace_id,
        "user_id": state.get("user_id", "00000000-0000-0000-0000-000000000000"),
        "workspace_id": state.get("workspace_id", "00000000-0000-0000-0000-000000000000"),
        "raw_prompt": state.get("raw_prompt", ""),
        "raw_prompt_hash": _prompt_hash_bytes(state.get("raw_prompt", "")),
        "raw_prompt_emb": embedding if (embedding and any(embedding)) else None,
        "project_context": _j(state.get("project_context")),
        "taxonomy_version": "v15",
        "fast_intent": _j(state.get("fast_intent")),
        "classification": _j(state.get("task_classification")),
        "scores": _j(scores),
        "clarifications": _j(state.get("clarifications")),
        "master_prompt": master.synthesized_prompt if master else "",
        "workflow_plan": _j(state.get("workflow_plan")),
        "routing_decision": _j(routing),
        "outcome": _j(outcome),
        "feedback": _j(state.get("feedback")),
        "fleet_dag": _j(state.get("fleet_dag")),
        "extracted_experience": _j(state.get("extracted_experience")),
        "model_tier_decisions": _j(state.get("model_tier_decisions")),
        "critical_path": state.get("critical_path", ""),
        "cost_usd": 0.0,
        "tokens_in": state.get("obs_tokens", 0) + state.get("memory_tokens", 0),
        "tokens_out": state.get("action_tokens", 0),
        "wall_ms": sum(e.wall_ms for e in state.get("execution_trace", [])),
        "retry_count": state.get("clarification_round", 0),
    }

    try:
        from backend.db.client import db_client
        await db_client.insert_trace(trace_record)
        log.info("session_persister.trace_written", trace_id=trace_id)
    except Exception as exc:
        log.error("session_persister.trace_error", error=str(exc))

    # ── 1b. Persist all routing decisions (active + shadows) ──────────────────
    shadow_rows = state.get("_shadow_routing_rows", [])  # type: ignore[typeddict-item]
    if shadow_rows:
        try:
            from backend.db.client import db_client
            await db_client.insert_routing_decisions(trace_id, shadow_rows)
            log.info(
                "session_persister.routing_decisions_written",
                trace_id=trace_id,
                n=len(shadow_rows),
            )
        except Exception as exc:
            log.warning("session_persister.routing_decisions_error", error=str(exc))

    # ── 1c. Persist outcome row (separate from trace.outcome JSON) ────────────
    if outcome is not None:
        try:
            from backend.db.client import db_client
            await db_client.insert_outcome(
                trace_id=trace_id,
                outcome=outcome.model_dump() if hasattr(outcome, "model_dump") else outcome,
            )
        except Exception as exc:
            log.warning("session_persister.outcome_error", error=str(exc))

    # ── 2. Upsert session row for multi-turn continuity ───────────────────────
    session_id = state.get("_session_id")  # type: ignore[typeddict-item]
    if not session_id:
        session_id = str(uuid.uuid4())
        state["_session_id"] = session_id  # type: ignore[typeddict-unknown-key]

    q_score = quality.score if quality else 0.0
    q_history_entry = q_score

    # Build cumulative context delta for this turn
    ctx = state.get("project_context")
    cumulative_delta: dict[str, Any] = {}
    if ctx:
        cumulative_delta["language"] = ctx.language
        cumulative_delta["framework"] = ctx.framework
        if ctx.claude_md_summary:
            cumulative_delta["claude_md_summary"] = ctx.claude_md_summary
    if routing:
        cumulative_delta["last_model"] = routing.chosen_model
    if state.get("task_classification"):
        cumulative_delta["last_task_type"] = state["task_classification"].task_type.value  # type: ignore

    token_consumed = (
        state.get("obs_tokens", 0)
        + state.get("memory_tokens", 0)
        + state.get("plan_tokens", 0)
        + state.get("action_tokens", 0)
    )

    try:
        from backend.db.client import db_client
        await db_client.upsert_session(
            session_id=session_id,
            user_id=state.get("user_id", ""),
            workspace_id=state.get("workspace_id", ""),
            original_prompt=state.get("raw_prompt", ""),
            cumulative_delta=cumulative_delta,
            token_budget_consumed=token_consumed,
            q_score=q_history_entry,
            fleet_state=_j(state.get("fleet_spec")),
        )
        log.info("session_persister.session_upserted", session_id=session_id)
    except Exception as exc:
        log.error("session_persister.session_error", error=str(exc))

    # ── 3. Update Redis cache on accepted outcome ─────────────────────────────
    try:
        import redis.asyncio as aioredis
        from backend.config import settings

        user_accepted = getattr(outcome, "user_accepted", None)
        if routing and master and user_accepted is True:
            cache_entry = {
                "trace_id": trace_id,
                "master_prompt": master.synthesized_prompt,
                "plan": json.dumps(state.get("workflow_plan", {}).steps if state.get("workflow_plan") else []),
                "routing_decision": routing.model_dump(),
            }
            prompt_hash = state.get("_prompt_hash", trace_id)  # type: ignore[typeddict-item]
            workspace_id = state.get("workspace_id", "")
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            async with r:
                await r.setex(
                    f"cache:exact:{prompt_hash}:{workspace_id}",
                    3600,
                    json.dumps(cache_entry),
                )
    except Exception as exc:
        log.warning("session_persister.cache_error", error=str(exc))

    # ── 4. Langfuse quality score ─────────────────────────────────────────────
    try:
        from backend.observability import emit_trace_score
        emit_trace_score(trace_id=trace_id, quality_score=q_score)
    except Exception:
        pass

    # ── 5. D-LinUCB bandit update (§5.3.12) ───────────────────────────────────
    try:
        arm_id = state.get("bandit_arm_selected")  # type: ignore[typeddict-item]
        x_ctx = state.get("_bandit_context_vector")  # type: ignore[typeddict-item]
        if arm_id and x_ctx:
            import numpy as np
            from backend.routing.bandit import BanditEnsemble, compute_reward

            workspace_id = state.get("workspace_id", "")
            ensemble = await BanditEnsemble.load(workspace_id)

            outcome_obj = outcome
            retries = state.get("clarification_round", 0) + (
                state.get("_replan_count", 0) or 0  # type: ignore[typeddict-item]
            )
            tokens_total = state.get("obs_tokens", 0) + state.get("action_tokens", 0)
            wall_ms = sum(e.wall_ms for e in state.get("execution_trace", []))

            reward = compute_reward(
                quality=q_score,
                monetary_cost_norm=0.0,
                latency_norm=min(1.0, wall_ms / 60000.0),
                retries=retries,
                token_waste_norm=min(1.0, tokens_total / 50000.0),
                first_try_success=(retries == 0 and q_score >= 0.75),
            )
            ensemble.get(arm_id).update(np.array(x_ctx, dtype=np.float64), reward)
            await ensemble.save_arm(workspace_id, arm_id)
            log.info(
                "session_persister.bandit_updated",
                arm=arm_id, reward=round(reward, 3),
            )
    except Exception as exc:
        log.warning("session_persister.bandit_update_error", error=str(exc))

    return state
