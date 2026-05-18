"""
LangGraph StateGraph — all 17 pipeline agents (§5, §6.2).

Execution order (Figure 2):
  1  prompt_intake
  2  session_loader          ← cache short-circuit here
  3+4 classify_and_score
  5  decide_clarification
  6  generate_clarifications  ─┐
  7  sufficiency_qa            ├ clarification loop
  8  re_scorer                ─┘
  9  synthesize_master_prompt
  10+11 route_and_resolve
  12 fleet_generator
  13 workflow_visualizer
  14 execute_via_claude_code
  15 quality_monitor
  16 adaptation_engine
  17 session_persister
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.pipeline.state import OptiviaState
from backend.pipeline.nodes.prompt_intake import prompt_intake
from backend.pipeline.nodes.session_loader import session_loader
from backend.pipeline.nodes.experience_retriever import experience_retriever
from backend.pipeline.nodes.cache_lookup import cache_lookup          # Tier-0 exact hash (kept for compatibility)
from backend.pipeline.nodes.fast_intent import fast_intent
from backend.pipeline.nodes.classify_score import classify_and_score
from backend.pipeline.nodes.clarification import (
    decide_clarification,
    generate_clarifications,
    sufficiency_qa,
    re_scorer,
)
from backend.pipeline.nodes.synthesize import synthesize_master_prompt
from backend.pipeline.nodes.route import route_and_resolve, fleet_generator
from backend.pipeline.nodes.visualizer import workflow_visualizer
from backend.pipeline.nodes.scheduler import critical_path_scheduler
from backend.pipeline.nodes.execution import execute_via_claude_code
from backend.pipeline.nodes.quality_monitor import quality_monitor, adaptation_engine
from backend.pipeline.nodes.experience_extractor import experience_extractor
from backend.pipeline.nodes.persist import session_persister

_MAX_REPLAN_LOOPS = 2


# ── Conditional edge predicates ───────────────────────────────────────────────

def _after_experience_retriever(state: OptiviaState) -> str:
    """Tier-0/1 cache hit → skip to persister (replay)."""
    if state.get("semantic_cache_hit"):
        return "replay"
    return "continue"


def _after_fast_intent(state: OptiviaState) -> str:
    fi = state.get("fast_intent")
    if fi and fi.short_circuit:
        return "short_circuit"
    return "continue"


def _after_clarification_decision(state: OptiviaState) -> str:
    if state.get("_clarify"):  # type: ignore[typeddict-item]
        return "clarify"
    return "skip"


def _after_sufficiency_qa(state: OptiviaState) -> str:
    if state.get("_sufficiency_passed"):  # type: ignore[typeddict-item]
        return "pass"
    if state.get("clarification_round", 0) >= 2:
        return "pass"   # fail open after max rounds
    return "fail"


def _after_quality(state: OptiviaState) -> str:
    branch = state.get("quality_branch", "pass")
    # Guard against infinite replan loops
    replan_count = state.get("_replan_count", 0)  # type: ignore[typeddict-item]
    if branch == "halt" and replan_count < _MAX_REPLAN_LOOPS:
        return "replan"
    return "continue"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(OptiviaState)

    # Register all pipeline agents as nodes
    g.add_node("prompt_intake", prompt_intake)            # 1
    g.add_node("session_loader", session_loader)          # 2
    g.add_node("experience_retriever", experience_retriever) # 2B
    g.add_node("fast_intent", fast_intent)                # 2b (triage)
    g.add_node("classify_and_score", classify_and_score)  # 3+4
    g.add_node("decide_clarification", decide_clarification)        # 5
    g.add_node("generate_clarifications", generate_clarifications)  # 6
    g.add_node("sufficiency_qa", sufficiency_qa)          # 7
    g.add_node("re_scorer", re_scorer)                    # 8
    g.add_node("synthesize_master_prompt", synthesize_master_prompt)  # 9
    g.add_node("route_and_resolve", route_and_resolve)    # 10+11
    g.add_node("fleet_generator", fleet_generator)        # 12
    g.add_node("workflow_visualizer", workflow_visualizer)  # 13A
    g.add_node("critical_path_scheduler", critical_path_scheduler) # 13B
    g.add_node("execute_via_claude_code", execute_via_claude_code)  # 14
    g.add_node("quality_monitor", quality_monitor)        # 15
    g.add_node("experience_extractor", experience_extractor) # 15B
    g.add_node("adaptation_engine", adaptation_engine)    # 16
    g.add_node("session_persister", session_persister)    # 17

    # ── Entry ────────────────────────────────────────────────────────────────
    g.add_edge(START, "prompt_intake")
    g.add_edge("prompt_intake", "session_loader")
    g.add_edge("session_loader", "experience_retriever")

    # Cache hit short-circuit after retrieving experience
    g.add_conditional_edges("experience_retriever", _after_experience_retriever, {
        "replay": "session_persister",
        "continue": "fast_intent",
    })

    # Fast intent triage
    g.add_conditional_edges("fast_intent", _after_fast_intent, {
        "short_circuit": "session_persister",
        "continue": "classify_and_score",
    })

    # Classify + score → clarification gate
    g.add_edge("classify_and_score", "decide_clarification")

    # Clarification branch
    g.add_conditional_edges("decide_clarification", _after_clarification_decision, {
        "clarify": "generate_clarifications",
        "skip": "synthesize_master_prompt",
    })

    # Clarification loop: 6 → 7 → (fail → 6 | pass → 8)
    g.add_edge("generate_clarifications", "sufficiency_qa")
    g.add_conditional_edges("sufficiency_qa", _after_sufficiency_qa, {
        "fail": "generate_clarifications",
        "pass": "re_scorer",
    })
    g.add_edge("re_scorer", "synthesize_master_prompt")

    # Main path: 9 → 10/11 → 12 → 13A → 13B → 14 → 15 → 15B → 16
    g.add_edge("synthesize_master_prompt", "route_and_resolve")
    g.add_edge("route_and_resolve", "fleet_generator")
    g.add_edge("fleet_generator", "workflow_visualizer")
    g.add_edge("workflow_visualizer", "critical_path_scheduler")
    g.add_edge("critical_path_scheduler", "execute_via_claude_code")
    g.add_edge("execute_via_claude_code", "quality_monitor")
    g.add_edge("quality_monitor", "experience_extractor")
    g.add_edge("experience_extractor", "adaptation_engine")

    # Quality branch: halt → replan (back to classify); otherwise → persist
    g.add_conditional_edges("adaptation_engine", _after_quality, {
        "replan": "classify_and_score",
        "continue": "session_persister",
    })

    g.add_edge("session_persister", END)

    return g


pipeline = build_graph().compile()
