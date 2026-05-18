"""HeuristicRouter — the always-available baseline (§4.4)."""

from __future__ import annotations

from backend.core.math_core import make_routing_decision
from backend.core.models import RoutingDecision
from backend.routing.base import Router, RoutingContext


class HeuristicRouter:
    """Hard rules driven by complexity κ and risk δ_r. The default Stage 1 router."""

    name = "heuristic_v1"

    async def route(self, ctx: RoutingContext) -> RoutingDecision:
        decision = make_routing_decision(
            scores=ctx.scores,
            task_type=ctx.task_classification.task_type,
            slash_commands=[],
        )
        # Tag the decision so the shadow log knows which router produced it
        decision.router_name = self.name
        return decision
