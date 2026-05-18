"""Tests for the routing module (§4.4) — three swappable routers + shadow runner."""

import pytest

from backend.core.models import TaskClassification, TaskScores, TaskType
from backend.core.math_core import enrich_scores
from backend.config import settings
from backend.routing import (
    HeuristicRouter,
    RouteLLMRouter,
    ShadowRouterRunner,
    RoutingContext,
)


def _ctx(**kw) -> RoutingContext:
    defaults = dict(scope=0.3, ambiguity=0.3, risk=0.2, dependency=0.2, context_load=0.2, confidence=0.8)
    defaults.update(kw)
    scores = enrich_scores(TaskScores(**defaults))
    return RoutingContext(
        raw_prompt="fix the bug",
        task_classification=TaskClassification(task_type=TaskType.DEBUG),
        scores=scores,
    )


# ── HeuristicRouter ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_heuristic_low_complexity_picks_haiku():
    ctx = _ctx(scope=0, ambiguity=0, risk=0, dependency=0, context_load=0)
    d = await HeuristicRouter().route(ctx)
    assert d.chosen_model == settings.model_haiku
    assert d.router_name == "heuristic_v1"


@pytest.mark.asyncio
async def test_heuristic_high_complexity_picks_opus():
    ctx = _ctx(scope=1, ambiguity=1, risk=1, dependency=1, context_load=1)
    d = await HeuristicRouter().route(ctx)
    assert d.chosen_model == settings.model_opus


@pytest.mark.asyncio
async def test_heuristic_midrange_picks_sonnet():
    # κ ~ 6 → sonnet_balanced
    ctx = _ctx(scope=0.5, ambiguity=0.5, risk=0.5, dependency=0.5, context_load=0.5)
    d = await HeuristicRouter().route(ctx)
    assert d.chosen_model == settings.model_sonnet


# ── RouteLLMRouter ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_routellm_trivial_picks_haiku():
    ctx = _ctx(scope=0.05, ambiguity=0.05, risk=0.05, dependency=0.05, context_load=0.05)
    d = await RouteLLMRouter().route(ctx)
    assert d.chosen_model == settings.model_haiku
    assert d.router_name == "routellm_mf_v1"
    # alternatives should contain the other two models with their win-rates
    assert len(d.alternatives) == 2


@pytest.mark.asyncio
async def test_routellm_high_risk_avoids_haiku():
    """Policy layer: never route risk ≥ 0.7 to Haiku."""
    ctx = _ctx(scope=0.1, ambiguity=0.1, risk=0.9, dependency=0.1, context_load=0.1)
    d = await RouteLLMRouter().route(ctx)
    assert d.chosen_model != settings.model_haiku


@pytest.mark.asyncio
async def test_routellm_returns_router_score():
    ctx = _ctx()
    d = await RouteLLMRouter().route(ctx)
    assert 0 <= d.router_score <= 1


# ── ShadowRouterRunner ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shadow_runner_returns_active_decision():
    """Heuristic is the active router by default; it must be the one acted on."""
    runner = ShadowRouterRunner(routers=[HeuristicRouter(), RouteLLMRouter()])
    ctx = _ctx()
    active, shadow = await runner.run(ctx)
    assert active is not None
    # Exactly one shadow row should be flagged was_active
    actives = [r for r in shadow if r["was_active"]]
    assert len(actives) == 1
    assert actives[0]["router_name"] == "heuristic_v1"


@pytest.mark.asyncio
async def test_shadow_runner_logs_all_routers():
    """Every router must produce a shadow row for Stage 2 training."""
    runner = ShadowRouterRunner(routers=[HeuristicRouter(), RouteLLMRouter()])
    ctx = _ctx()
    _, shadow = await runner.run(ctx)
    assert len(shadow) == 2
    names = {r["router_name"] for r in shadow}
    assert "heuristic_v1" in names
    assert "routellm_mf_v1" in names


@pytest.mark.asyncio
async def test_shadow_runner_captures_features():
    """Every shadow row needs router_features for Stage 2 training."""
    runner = ShadowRouterRunner(routers=[HeuristicRouter()])
    ctx = _ctx(scope=0.6, ambiguity=0.4, risk=0.5)
    _, shadow = await runner.run(ctx)
    feats = shadow[0]["router_features"]
    assert feats["scope"] == 0.6
    assert feats["risk"] == 0.5
    assert "complexity" in feats
