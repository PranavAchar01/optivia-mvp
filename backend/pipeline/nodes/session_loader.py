"""
Agent 2 — Session Loader (§5, §10.1).

Responsibilities:
- Load persistent SessionState from Postgres for multi-turn continuity
- Inject cumulative_context (prior framework decisions, file diffs, Q_history)
- Seed token_budget_consumed from prior turns
- Short-circuit semantic cache via Redis exact-hash (Tier 0) and embedding (Tier 1)
"""

from __future__ import annotations

import hashlib
import json

import redis.asyncio as aioredis
import structlog

from backend.config import settings
from backend.core.models import CachedResult, RoutingDecision
from backend.embeddings import embed
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)


async def session_loader(state: OptiviaState) -> OptiviaState:
    """Agent 2 — Session Loader."""
    raw = state.get("raw_prompt", "")
    session_id = state.get("_session_id")  # type: ignore[typeddict-item]
    workspace_id = state.get("workspace_id", "")
    
    # Simple heuristic: ~4 chars per token for observation token tracking
    state["obs_tokens"] = state.get("obs_tokens", 0) + len(raw) // 4

    # ── Tier 0: exact hash cache ─────────────────────────────────────────────
    prompt_hash = hashlib.sha256(raw.encode()).hexdigest()
    state["_prompt_hash"] = prompt_hash  # type: ignore[typeddict-unknown-key]

    try:
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        async with r:
            exact = await r.get(f"cache:exact:{prompt_hash}:{workspace_id}")
            if exact:
                try:
                    data = json.loads(exact)
                    state["semantic_cache_hit"] = CachedResult(
                        trace_id=data.get("trace_id", ""),
                        master_prompt=data.get("master_prompt", ""),
                        plan=data.get("plan", ""),
                        routing_decision=RoutingDecision(**data.get("routing_decision", {})),
                        similarity=1.0,
                    )
                    log.info("session_loader.cache_exact_hit", request_id=state.get("request_id"))
                    return state
                except json.JSONDecodeError:
                    log.warning("session_loader.cache_exact_corrupt", prompt_hash=prompt_hash)
    except Exception as exc:
        log.warning("session_loader.cache_error", error=str(exc))

    # ── Tier 1: semantic cache via pgvector (§4.5) ────────────────────────────
    try:
        embedding = await embed(raw, input_type="query")
        # Stash the embedding so the persister can write it without re-embedding
        state["_prompt_embedding"] = embedding  # type: ignore[typeddict-unknown-key]

        if any(embedding):  # skip if voyage call failed (all zeros)
            from backend.db.client import db_client
            hit = await db_client.find_semantic_cache(
                embedding=embedding,
                workspace_id=workspace_id,
                threshold=settings.semantic_cache_threshold,
            )
            if hit:
                routing_payload = hit.get("routing_decision") or {}
                if isinstance(routing_payload, str):
                    try:
                        routing_payload = json.loads(routing_payload)
                    except json.JSONDecodeError:
                        routing_payload = {}
                try:
                    routing_decision = RoutingDecision(**routing_payload)
                except Exception:
                    routing_decision = RoutingDecision(chosen_model=settings.model_sonnet)

                state["semantic_cache_hit"] = CachedResult(
                    trace_id=str(hit["id"]),
                    master_prompt=hit.get("master_prompt", ""),
                    plan=json.dumps(hit.get("workflow_plan", {})),
                    routing_decision=routing_decision,
                    similarity=float(hit.get("similarity") or 0.0),
                )
                log.info(
                    "session_loader.semantic_cache_hit",
                    similarity=hit.get("similarity"),
                    trace_id=str(hit["id"]),
                )
                return state
    except Exception as exc:
        log.warning("session_loader.semantic_cache_error", error=str(exc))

    # ── Load multi-turn session from Postgres ─────────────────────────────────
    if session_id:
        try:
            from backend.db.client import db_client
            row = await db_client.get_session(session_id)
            if row:
                state["memory_tokens"] = state.get("memory_tokens", 0) + row.get("token_budget_consumed", 0)

                # Inject cumulative context so scorer/re-scorer sees prior decisions
                cumulative = row.get("cumulative_context", {})
                if cumulative:
                    ctx = state.get("project_context")
                    if ctx and not ctx.claude_md_summary and cumulative.get("claude_md_summary"):
                        ctx.claude_md_summary = cumulative["claude_md_summary"]
                        state["project_context"] = ctx

                # Seed Q_history for long-clean-stretch detection
                q_hist = row.get("q_history", [])
                if isinstance(q_hist, list) and q_hist:
                    consecutive = sum(1 for q in reversed(q_hist) if q > 0.9)
                    state["consecutive_high_quality"] = consecutive

                log.info(
                    "session_loader.session_resumed",
                    session_id=session_id,
                    turn_index=state.get("turn_index", 0),
                )
        except Exception as exc:
            log.warning("session_loader.session_load_error", error=str(exc))

    return state
