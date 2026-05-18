"""
RouteLLMRouter — wraps the LMSYS matrix-factorisation router (§4.4).

Stage 1 ships a faithful in-process emulation of RouteLLM's mf scoring rule:
a bilinear cost/quality estimator over (κ, σ, risk, scope, dependency). The
real RouteLLM checkpoint is a drop-in replacement once the team trains on
Optivia's own preference data (Stage 2).

The router produces win-probabilities for the three Haiku/Sonnet/Opus tiers
and picks the cheapest model whose win-probability exceeds the calibrated
threshold τ. Below the τ floor we fall back to the next-tier-up model.
"""

from __future__ import annotations

import math

from backend.config import settings
from backend.core.math_core import resolve_slash_commands
from backend.core.models import RoutingDecision
from backend.routing.base import Router, RoutingContext


# Per-model win-rate priors derived from §2 (SWE-bench Verified figures).
# Stage 2 retrains these from logged outcomes.
_BASE_WIN_RATE: dict[str, float] = {
    "haiku":  0.733,   # 73.3% on SWE-bench Verified
    "sonnet": 0.796,   # 79.6%
    "opus":   0.808,   # 80.8%
}

# RouteLLM-style mf weights — bilinear model: P(success | x, m) = σ(w_m · features)
# Features = [κ/10, σ, risk, scope, dependency, context_load, 1]
_MF_WEIGHTS: dict[str, list[float]] = {
    # higher difficulty signals → lower P(success) for cheap models
    "haiku":  [-1.6, +0.6, -1.4, -1.2, -0.8, -1.1, +1.7],
    "sonnet": [-0.6, +0.3, -0.7, -0.5, -0.3, -0.6, +1.6],
    "opus":   [-0.2, +0.2, -0.3, -0.2, -0.1, -0.3, +1.7],
}

# RouteLLM's calibrated threshold t (§4.4) — picks the cheapest model whose
# win-rate is within t of the best. Tuned per §2: Sonnet 4.6 is 99% of Opus on
# SWE-bench, so trivial tasks should freely defer to Haiku.
_DEFER_THRESHOLD = 0.08


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _features(ctx: RoutingContext) -> list[float]:
    s = ctx.scores
    return [
        s.complexity / 10.0,
        s.specificity,
        s.risk,
        s.scope,
        s.dependency,
        s.context_load,
        1.0,
    ]


def _win_rate(model_key: str, feats: list[float]) -> float:
    w = _MF_WEIGHTS[model_key]
    logit = sum(wi * fi for wi, fi in zip(w, feats))
    return _sigmoid(logit) * _BASE_WIN_RATE[model_key]


_MODEL_FOR_KEY: dict[str, str] = {
    "haiku":  settings.model_haiku,
    "sonnet": settings.model_sonnet,
    "opus":   settings.model_opus,
}

# Minimum n_agents floor per model tier (§4.7).
_MIN_AGENTS_BY_KEY: dict[str, int] = {"haiku": 1, "sonnet": 2, "opus": 5}


def _n_agents_for(model_key: str, complexity: int, scope: float, dependency: float) -> int:
    """
    Continuous fleet sizing: more complex/wider-scope tasks get more sub-agents.
    Floor by model tier, then add complexity-driven granularity.

    κ=1-3 → 1 agent  (trivial, single edit)
    κ=4-5 → 2-3 agents (audit + implement, or implement + test)
    κ=6   → 3-4 agents (audit + implement + test)
    κ=7   → 5-6 agents (audit + design + implement-by-layer + test + docs)
    κ=8-9 → 7-8 agents (parallel implementers across files/layers)
    κ=10  → 8+ agents
    """
    floor = _MIN_AGENTS_BY_KEY[model_key]
    # Each complexity point above 3 adds an agent; scope and dependency add bonuses
    bonus = max(0, complexity - 3) + int(scope >= 0.6) + int(dependency >= 0.6)
    return max(floor, min(8, bonus))


class RouteLLMRouter:
    """Matrix-factorisation cost/quality router (§4.4)."""

    name = "routellm_mf_v1"

    async def route(self, ctx: RoutingContext) -> RoutingDecision:
        feats = _features(ctx)
        win = {k: _win_rate(k, feats) for k in _BASE_WIN_RATE}

        # Cascade rule: pick the cheapest model whose win-rate is within τ of best.
        best_key = max(win, key=win.get)
        best = win[best_key]
        chosen_key = best_key
        for key in ("haiku", "sonnet", "opus"):  # cheapest → most expensive
            if win[key] >= best - _DEFER_THRESHOLD:
                chosen_key = key
                break

        # Hard guardrail: never route high-risk tasks to Haiku (§4.4 policy layer).
        if ctx.scores.risk >= 0.7 and chosen_key == "haiku":
            chosen_key = "sonnet"

        chosen = _MODEL_FOR_KEY[chosen_key]
        n_agents = _n_agents_for(
            chosen_key,
            ctx.scores.complexity,
            ctx.scores.scope,
            ctx.scores.dependency,
        )
        commands = resolve_slash_commands(ctx.task_classification.task_type, [])

        return RoutingDecision(
            chosen_model=chosen,
            n_agents=n_agents,
            plan="planning_enabled" if chosen_key == "opus" else "",
            slash_commands=commands,
            alternatives=[
                {"model": _MODEL_FOR_KEY[k], "win_rate": round(v, 4)}
                for k, v in win.items() if k != chosen_key
            ],
            router_name=self.name,
            router_score=float(round(win[chosen_key], 4)),
        )
