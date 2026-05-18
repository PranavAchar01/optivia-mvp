"""Integration-style tests for agents 1, 2, 5, 13, 16, 17 (no I/O mocked)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.models import ProjectContext, RoutingDecision, TaskClassification, TaskScores, TaskType, WorkflowPlan
from backend.pipeline.nodes.prompt_intake import prompt_intake, _detect_language, _detect_framework
from backend.pipeline.nodes.fast_intent import fast_intent, _classify_fast
from backend.pipeline.nodes.clarification import decide_clarification
from backend.pipeline.nodes.visualizer import workflow_visualizer, _build_react_flow_graph
from backend.pipeline.nodes.quality_monitor import adaptation_engine
from backend.core.math_core import enrich_scores


# ── Agent 1: Prompt Intake ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prompt_intake_assigns_ids():
    state = {"raw_prompt": "build a login system"}
    result = await prompt_intake(state)
    assert result.get("request_id")
    assert result.get("workspace_id")
    assert result.get("project_context") is not None


@pytest.mark.asyncio
async def test_prompt_intake_empty_prompt():
    state = {"raw_prompt": ""}
    result = await prompt_intake(state)
    assert result.get("error") == "empty prompt"


@pytest.mark.asyncio
async def test_prompt_intake_parses_file_refs(tmp_path):
    (tmp_path / "auth.py").write_text("def login(): pass")
    state = {
        "raw_prompt": f"fix the bug in @auth.py",
        "project_context": ProjectContext(repo_root=str(tmp_path)),
    }
    result = await prompt_intake(state)
    files = result.get("attached_files", [])
    assert any("auth.py" in f.path for f in files)


def test_detect_language_python(tmp_path):
    for i in range(5):
        (tmp_path / f"mod{i}.py").write_text("pass")
    lang = _detect_language(str(tmp_path))
    assert lang == "Python"


def test_detect_framework_nextjs(tmp_path):
    (tmp_path / "package.json").write_text('{"dependencies": {"next": "15.0"}}')
    assert _detect_framework(str(tmp_path)) == "Next.js"


def test_detect_framework_fastapi(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="app"')
    assert _detect_framework(str(tmp_path)) == "FastAPI/Python"


# ── Agent 2: Fast Intent ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fast_intent_code_task():
    state = {"raw_prompt": "build and implement a new auth system with JWT tokens"}
    result = await fast_intent(state)
    fi = result.get("fast_intent")
    assert fi is not None
    assert fi.intent == "code_task"
    assert not fi.short_circuit


@pytest.mark.asyncio
async def test_fast_intent_short_circuits_chitchat():
    state = {"raw_prompt": "hi"}
    result = await fast_intent(state)
    fi = result.get("fast_intent")
    assert fi is not None
    assert fi.short_circuit is True


def test_classify_fast_multiple_keywords():
    intent, conf = _classify_fast("fix and debug the authentication bug in the login module")
    assert intent == "code_task"
    assert conf >= 0.8


# ── Agent 5: Clarification Trigger ───────────────────────────────────────────

def test_decide_clarification_fires_max_kappa():
    scores = TaskScores(scope=0, ambiguity=0, risk=0, dependency=0, context_load=0, confidence=0.9)
    scores.complexity = 10
    scores.specificity = 1.0
    state = {
        "scores": scores,
        "clarification_round": 0,
        "_clarify": False,
    }
    result = decide_clarification(state)
    assert result["_clarify"] is True


def test_decide_clarification_skips_low_kappa():
    scores = enrich_scores(TaskScores(scope=0.1, ambiguity=0.1, risk=0.1, dependency=0.1, context_load=0.1))
    state = {"scores": scores, "clarification_round": 0}
    result = decide_clarification(state)
    assert result.get("_clarify") is False


def test_decide_clarification_respects_max_rounds():
    scores = TaskScores(scope=1, ambiguity=1, risk=1, dependency=1, context_load=1)
    scores.complexity = 10
    scores.specificity = 0.0
    state = {"scores": scores, "clarification_round": 2}
    result = decide_clarification(state)
    assert result.get("_clarify") is False


# ── Agent 13: Workflow Visualizer ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_workflow_visualizer_produces_graph():
    routing = RoutingDecision(chosen_model="claude-sonnet-4-6", n_agents=2)
    state = {
        "workflow_plan": WorkflowPlan(steps=["Scaffold auth module", "Write tests", "Review diff"]),
        "fleet_spec": {"roles": ["scaffolder", "tester"], "n_agents": 2},
        "task_classification": TaskClassification(task_type=TaskType.NEW_CODE),
        "routing_decision": routing,
    }
    result = await workflow_visualizer(state)
    plan = result.get("workflow_plan")
    assert plan is not None
    graph = plan.visualizer_json
    assert len(graph["nodes"]) == 2   # 1 node per role
    assert len(graph["edges"]) == 1   # edges between roles


def test_build_react_flow_single_step():
    routing = RoutingDecision(chosen_model="claude-haiku-4-5", n_agents=1)
    graph = _build_react_flow_graph({"roles": ["executor"], "n_agents": 1}, "trivial", routing)
    # 1 agent node
    assert len(graph["nodes"]) == 1
    assert len(graph["edges"]) == 0


def test_build_react_flow_accent_color():
    routing = RoutingDecision(chosen_model="claude-sonnet-4-6", n_agents=1)
    graph = _build_react_flow_graph({}, "debug", routing)
    assert graph["accent"] == "#f59e0b"


# ── Agent 16: Adaptation Engine ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_adaptation_engine_halt_triggers_replan():
    state = {
        "quality_branch": "halt",
        "token_budget_action": "ok",
        "adaptation_actions": [],
    }
    result = await adaptation_engine(state)
    assert "replan_via_scorer" in result.get("adaptation_actions", [])


@pytest.mark.asyncio
async def test_adaptation_engine_budget_switches_model():
    routing = RoutingDecision(chosen_model="claude-opus-4-6", n_agents=1)
    state = {
        "quality_branch": "pass",
        "token_budget_action": "switch_model",
        "routing_decision": routing,
        "adaptation_actions": [],
    }
    result = await adaptation_engine(state)
    assert "downgrade_to_haiku" in result.get("adaptation_actions", [])
    from backend.config import settings
    assert result["routing_decision"].chosen_model == settings.model_haiku


@pytest.mark.asyncio
async def test_adaptation_engine_long_clean_stretch():
    state = {
        "quality_branch": "long_clean_stretch",
        "token_budget_action": "ok",
        "adaptation_actions": [],
    }
    result = await adaptation_engine(state)
    assert "relax_verification_overhead" in result.get("adaptation_actions", [])
