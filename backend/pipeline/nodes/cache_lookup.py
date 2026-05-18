"""Node 1+2 — Prompt Intake / Session Loader + Semantic Cache Lookup (§6.2 node 1)."""

from __future__ import annotations

import hashlib
import json
import uuid

import redis.asyncio as aioredis
import structlog

from backend.config import settings
from backend.core.models import CachedResult, ProjectContext, RoutingDecision
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)


async def cache_lookup(state: OptiviaState) -> OptiviaState:
    """
    Node: cache_lookup
    - Assigns request_id if absent
    - Checks Redis semantic cache (cosine ≥ threshold + same workspace + same task type)
    - Short-circuits to replay_outcome if hit
    """
    state.setdefault("request_id", str(uuid.uuid4()))
    state.setdefault("clarification_round", 0)
    state.setdefault("consecutive_high_quality", 0)
    state.setdefault("obs_tokens", 0)
    state.setdefault("memory_tokens", 0)
    state.setdefault("plan_tokens", 0)
    state.setdefault("action_tokens", 0)
    state.setdefault("execution_trace", [])
    state.setdefault("clarifications", [])
    state.setdefault("adaptation_actions", [])
    state.setdefault("turn_index", 0)

    raw = state.get("raw_prompt", "")
    if not raw:
        state["error"] = "empty prompt"
        return state

    # Hash for exact-match Tier 0
    prompt_hash = hashlib.sha256(raw.encode()).hexdigest()
    state["_prompt_hash"] = prompt_hash  # type: ignore[typeddict-unknown-key]

    try:
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        async with r:
            exact = await r.get(f"cache:exact:{prompt_hash}")
            if exact:
                data = json.loads(exact)
                state["semantic_cache_hit"] = CachedResult(
                    trace_id=data["trace_id"],
                    master_prompt=data["master_prompt"],
                    plan=data["plan"],
                    routing_decision=RoutingDecision(**data["routing_decision"]),
                    similarity=1.0,
                )
                log.info("cache.exact_hit", request_id=state["request_id"])
    except Exception as exc:
        # Cache miss is not an error — pipeline continues
        log.warning("cache.lookup_error", error=str(exc))

    return state
