"""
ShadowRouterRunner — runs every registered router on every request, returns
the active router's decision, and persists the rest as shadow logs (§4.4).

This is the linchpin of the Stage 1 trace contract: each routing_decisions
row gives Stage 2 a counterfactual it can train against. Without this, the
Stage 2 router has no preference data to learn from.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog

from backend.config import settings
from backend.core.models import RoutingDecision
from backend.routing.base import Router, RoutingContext
from backend.routing.heuristic import HeuristicRouter
from backend.routing.llm_judge import LLMJudgeRouter
from backend.routing.routellm import RouteLLMRouter

log = structlog.get_logger(__name__)


# Which router's decision is actually acted on. Stage 1 ships heuristic
# active because RouteLLM and LLMJudge are unproven on Optivia's distribution.
_DEFAULT_ACTIVE = "heuristic_v1"


class ShadowRouterRunner:
    """Runs N routers in parallel, returns the active decision, logs the rest."""

    def __init__(self, routers: list[Router] | None = None, active: str | None = None):
        self.routers: list[Router] = routers or [
            HeuristicRouter(),
            RouteLLMRouter(),
            LLMJudgeRouter(),
        ]
        self.active_name: str = active or _DEFAULT_ACTIVE

    async def run(self, ctx: RoutingContext) -> tuple[RoutingDecision, list[dict[str, Any]]]:
        """
        Returns (active_decision, shadow_rows) where shadow_rows is a list of
        dicts ready to write to the routing_decisions Postgres table.
        """
        # Run all routers concurrently; isolate failures so one bad router
        # doesn't take the request down.
        async def _run_one(router: Router) -> tuple[str, RoutingDecision | None, int]:
            t0 = time.monotonic()
            try:
                d = await router.route(ctx)
                return router.name, d, int((time.monotonic() - t0) * 1000)
            except Exception as exc:
                log.warning("shadow.router_error", router=router.name, error=str(exc))
                return router.name, None, int((time.monotonic() - t0) * 1000)

        results = await asyncio.gather(*[_run_one(r) for r in self.routers])

        active_decision: RoutingDecision | None = None
        shadow_rows: list[dict[str, Any]] = []

        for name, decision, elapsed_ms in results:
            if decision is None:
                continue
            is_active = name == self.active_name
            if is_active:
                active_decision = decision

            shadow_rows.append({
                "id": str(uuid.uuid4()),
                "router_name": name,
                "was_active": is_active,
                "chosen_model": decision.chosen_model,
                "alternatives": decision.alternatives,
                "router_features": {
                    "complexity": ctx.scores.complexity,
                    "specificity": ctx.scores.specificity,
                    "risk": ctx.scores.risk,
                    "scope": ctx.scores.scope,
                    "dependency": ctx.scores.dependency,
                    "context_load": ctx.scores.context_load,
                    "elapsed_ms": elapsed_ms,
                },
                "router_score": decision.router_score,
            })

        # Fallback: if the active router crashed, use the first non-None decision
        if active_decision is None:
            for _, d, _ in results:
                if d is not None:
                    active_decision = d
                    break

        if active_decision is None:
            # Total failure — emit a safe sonnet default so the pipeline keeps moving
            active_decision = RoutingDecision(
                chosen_model=settings.model_sonnet,
                n_agents=1,
                router_name="fallback",
            )

        return active_decision, shadow_rows


_runner: ShadowRouterRunner | None = None


def get_router_runner() -> ShadowRouterRunner:
    """Singleton — instantiated once per process."""
    global _runner
    if _runner is None:
        _runner = ShadowRouterRunner()
    return _runner
