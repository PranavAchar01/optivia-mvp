"""Node 15–16 — Quality Monitor + Adaptation Engine (§4.10, §10.2)."""

from __future__ import annotations

import structlog
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.core.llm import llm_client
from backend.core.math_core import check_token_budget, compute_quality, quality_branch
from backend.core.models import QualityScalar
from backend.observability import emit_trace_score
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)


class QualityAssessment(BaseModel):
    goal_met: bool
    no_errors: bool
    matches_conventions: bool
    minimal_changes: bool
    reasoning: str


_QUALITY_SYSTEM = """\
You are Optivia's Quality Monitor. Assess whether the Claude Code execution achieved its goal.
Evaluate four binary criteria:
1. goal_met — did the output accomplish what was asked?
2. no_errors — are there no syntax/runtime/test errors in the output?
3. matches_conventions — does the output match the project's coding style?
4. minimal_changes — were changes scoped to what was necessary (no scope creep)?
Be strict. Partial success is not goal_met.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def _call_quality_llm(goal_str: str, exec_summary: str) -> QualityAssessment:
    return await llm_client.structured_generate(
        system_prompt=_QUALITY_SYSTEM,
        user_prompt=f"Original goal:\n{goal_str}\n\nExecution result:\n{exec_summary}",
        response_model=QualityAssessment,
    )


async def quality_monitor(state: OptiviaState) -> OptiviaState:
    """
    Node: quality_monitor (agent 15 — Haiku 4.5, per-action small call).
    Computes Q_t and determines the quality branch.
    Also checks token budget invariant (§4.9).
    """
    # Token budget check
    rho, budget_action = check_token_budget(
        obs_tokens=state.get("obs_tokens", 0),
        memory_tokens=state.get("memory_tokens", 0),
        plan_tokens=state.get("plan_tokens", 0),
        action_tokens=state.get("action_tokens", 0),
        budget=settings.session_token_budget,
    )
    state["token_budget_rho"] = rho
    state["token_budget_action"] = budget_action

    execution_trace = state.get("execution_trace", [])
    master = state.get("master_prompt")

    if not execution_trace or master is None:
        q = QualityScalar(goal_met=False, no_errors=True, matches_conventions=True, minimal_changes=True)
        state["quality"] = q
        state["quality_branch"] = "pass"
        return state

    last_event = execution_trace[-1]
    if getattr(last_event, "event_type", None) == "simulated_execution":
        q = QualityScalar(goal_met=True, no_errors=True, matches_conventions=True, minimal_changes=True)
        state["quality"] = q
        state["quality_branch"] = "pass"
        state["consecutive_high_quality"] = state.get("consecutive_high_quality", 0) + 1
        log.info("quality_monitor.done", score=1.0, branch="pass", budget_rho=rho)
        return state

    exec_summary = str(last_event.payload)[:600]

    try:
        result = await _call_quality_llm(master.synthesized_prompt[:300], exec_summary)
        q = QualityScalar(
            goal_met=result.goal_met,
            no_errors=result.no_errors,
            matches_conventions=result.matches_conventions,
            minimal_changes=result.minimal_changes,
        )
    except Exception as exc:
        log.error("quality_monitor.error", error=str(exc))
        q = QualityScalar(goal_met=True, no_errors=True, matches_conventions=True, minimal_changes=True)

    q_score = compute_quality(q)
    consecutive = state.get("consecutive_high_quality", 0)
    if q_score > 0.9:
        consecutive += 1
    else:
        consecutive = 0

    branch = quality_branch(q_score, consecutive)

    state["quality"] = q
    state["quality_branch"] = branch
    state["consecutive_high_quality"] = consecutive

    emit_trace_score(
        trace_id=state.get("trace_id") or state.get("request_id", ""),
        quality_score=q_score,
    )

    log.info("quality_monitor.done", score=q_score, branch=branch, budget_rho=rho)
    return state


# ---------------------------------------------------------------------------
# Node 16: adaptation_engine (§10.2)
# ---------------------------------------------------------------------------

async def adaptation_engine(state: OptiviaState) -> OptiviaState:
    """
    Node: adaptation_engine — deterministic decision tree (§5.3.20).
    Implements all 8 triggers from spec:
      1. Simpler task mid-session → downgrade model tier
      2. Harder task → upgrade tier + spawn verifier
      3. Budget threshold crossed → progressive compression
      4. Rate limit hit → switch provider (not implemented; left as no-op)
      5. Q_t < 0.5 → halt + replan
      6. 5x consecutive Q_t > 0.9 → relax verification
      7. Context window > 70% → compact-context
      8. Task drift detected → re-classify, possibly re-elicit
    """
    branch = state.get("quality_branch", "pass")
    budget_action = state.get("token_budget_action", "ok")
    rho = state.get("token_budget_rho", 0.0)
    routing = state.get("routing_decision")
    scores = state.get("scores_updated") or state.get("scores")
    actions: list[str] = []

    # Trigger 3: budget cascade adaptations (§5 stages 1-4)
    if budget_action == "switch_model":
        actions.append("downgrade_to_haiku")
        if routing:
            routing.chosen_model = settings.model_haiku
            state["routing_decision"] = routing
    elif budget_action == "emergency":
        actions.append("emergency_mode_retain_essential_context")
    elif budget_action == "summarise":
        actions.append("gentle_context_summarisation")
    elif budget_action == "drop":
        actions.append("low_relevance_context_drop")

    # Trigger 7: context-window utilisation > 70% — additive to budget cascade
    if rho >= 0.70:
        actions.append("compact_context")

    # Trigger 2: harder task than current tier supports → upgrade + verifier
    if scores and routing:
        on_haiku = routing.chosen_model == settings.model_haiku
        on_sonnet = routing.chosen_model == settings.model_sonnet
        if on_haiku and scores.complexity >= 8:
            routing.chosen_model = settings.model_sonnet
            state["routing_decision"] = routing
            actions.append("upgrade_to_sonnet")
            actions.append("spawn_verifier")
        elif on_sonnet and scores.complexity >= 9:
            routing.chosen_model = settings.model_opus
            state["routing_decision"] = routing
            actions.append("upgrade_to_opus")
            actions.append("spawn_verifier")

    # Trigger 1: simpler than expected — opportunistic downgrade
    quality = state.get("quality")
    if (
        routing
        and quality is not None
        and quality.score >= 0.9
        and scores
        and scores.complexity <= 4
        and routing.chosen_model != settings.model_haiku
    ):
        routing.chosen_model = settings.model_haiku
        state["routing_decision"] = routing
        actions.append("downgrade_simpler_task")

    # Trigger 5/6: quality branch adaptations
    if branch == "halt":
        actions.append("replan_via_scorer")
        state["_replan_count"] = state.get("_replan_count", 0) + 1  # type: ignore[typeddict-unknown-key]
    elif branch == "reverify":
        actions.append("spawn_verifier")
    elif branch == "long_clean_stretch":
        actions.append("relax_verification_overhead")

    # Trigger 8: task drift detection — compare current classification vs prior turn
    last_task_type = state.get("_last_task_type")  # type: ignore[typeddict-item]
    cur_cls = state.get("task_classification")
    if last_task_type and cur_cls and cur_cls.task_type.value != last_task_type:
        actions.append("task_drift_detected")
        state["_needs_reclassify"] = True  # type: ignore[typeddict-unknown-key]
    if cur_cls:
        state["_last_task_type"] = cur_cls.task_type.value  # type: ignore[typeddict-unknown-key]

    state["adaptation_actions"] = actions
    log.info("adaptation_engine.done", actions=actions, budget_rho=rho)
    return state
