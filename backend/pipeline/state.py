"""OptiviaState — the single TypedDict shared across all LangGraph nodes (§6.2)."""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict

from backend.core.models import (
    CachedResult,
    Clarification,
    ExecutionEvent,
    ExperienceRecord,
    FastIntent,
    FileRef,
    MasterPrompt,
    Outcome,
    ProjectContext,
    QualityScalar,
    RoutingDecision,
    TaskClassification,
    TaskScores,
    UserFeedback,
    WorkflowPlan,
)


class OptiviaState(TypedDict, total=False):
    # Turn identity
    request_id: str
    user_id: str
    workspace_id: str
    turn_index: int

    # Input
    raw_prompt: str
    attached_files: list[FileRef]
    project_context: ProjectContext

    # Stage 1: cache short-circuit
    semantic_cache_hit: Optional[CachedResult]

    # Stage 2: fast intent triage
    fast_intent: FastIntent

    # Stage 3–4: classification + scoring
    task_classification: TaskClassification
    scores: TaskScores                   # v_initial after Scorer
    scores_updated: TaskScores           # v_updated after Re-Scorer (§4.6)

    # Stage 5–8: clarification loop
    clarifications: list[Clarification]
    clarification_round: int             # guards infinite loops

    # Stage 9: synthesis
    master_prompt: MasterPrompt

    # Stage 10–11: routing + conflict resolution
    routing_decision: RoutingDecision

    # Stage 12–13: fleet + visualisation
    workflow_plan: WorkflowPlan
    fleet_spec: dict                     # sub-agent roles / counts
    fleet_dag: dict                      # v15 JSON DAG output with Nodes and Edges
    critical_path: str                   # string representation of bottleneck warnings

    # Stage 14–15: execution + quality
    execution_trace: list[ExecutionEvent]
    quality: Optional[QualityScalar]
    quality_branch: str                  # "pass" | "halt" | "reverify" | "long_clean_stretch"
    consecutive_high_quality: int
    extracted_experience: list[str]      # one-sentence reusable lessons

    # Stage 16: adaptation
    adaptation_actions: list[str]

    # Stage 17: persistence
    trace_id: str
    outcome: Optional[Outcome]
    feedback: Optional[UserFeedback]
    
    # v15 Architecture specific states
    retrieved_lessons: list[ExperienceRecord]   # from Agent 2B (typed)
    model_tier_decisions: dict                   # from Agent 10B D-LinUCB Selector
    allowed_arms: list[str]                      # from Agent 10A safety filter
    bandit_arm_selected: str                     # chosen A0..A5 arm id
    cpl_norm: float                              # from Agent 13B (feeds bandit x_t)
    mega_prompt: str                             # from Agent 14 — full serialized fleet string

    # Token budget tracking (§4.9)
    obs_tokens: int
    memory_tokens: int
    plan_tokens: int
    action_tokens: int
    token_budget_rho: float
    token_budget_action: str             # "ok" | "summarise" | "drop" | ...

    # Error propagation
    error: Optional[str]
