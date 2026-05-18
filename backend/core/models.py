"""Pydantic models mirroring the Trace Contract (§6.3, §6.4)."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# §6.3 — Classification taxonomy (eight classes)
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    NEW_CODE = "new_code"    # G1 — build from scratch
    DEBUG = "debug"          # G2 — fix something broken
    REFACTOR = "refactor"    # G3 — change shape, not behaviour
    REVIEW = "review"        # G5 — read/explain/audit
    EXPLAIN = "explain"      # answer a question about code
    LONG = "long"            # G4 — multi-file, multi-step
    TRIVIAL = "trivial"      # rename, format, single-line (G∅)
    META = "meta"            # ask Claude Code itself


class TaskScores(BaseModel):
    """Five orthogonal [0,1]-normalised signals (§4.2)."""
    scope: float = Field(ge=0, le=1)        # δ_s — code area touched
    ambiguity: float = Field(ge=0, le=1)    # δ_a — how much unsaid
    risk: float = Field(ge=0, le=1)         # δ_r — blast radius
    dependency: float = Field(ge=0, le=1)   # δ_d — coupling
    context_load: float = Field(ge=0, le=1) # δ_c — files needed first

    # Derived fields (computed by the math core, not the LLM)
    complexity: int = Field(ge=1, le=10, default=1)   # κ_t
    specificity: float = Field(ge=0, le=1, default=1) # σ_t

    # Token/time estimates
    est_tokens_input: int = 0
    est_tokens_output: int = 0
    est_wall_seconds: int = 0
    est_tier: Literal["<15m", "15m-1h", "1h-4h", ">4h"] = "<15m"
    confidence: float = Field(ge=0, le=1, default=0.5)


class TaskClassification(BaseModel):
    task_type: TaskType
    taxonomy_version: str = "v1"
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Clarification
# ---------------------------------------------------------------------------

class ClarificationQuestion(BaseModel):
    dimension: str  # which signal the question targets (ambiguity / scope / risk / …)
    question: str
    expected_info_gain: float = Field(ge=0, le=1, default=0.5)


class Clarification(BaseModel):
    questions: list[ClarificationQuestion]
    answers: list[str] = Field(default_factory=list)
    sufficiency_passed: bool = False


# ---------------------------------------------------------------------------
# Routing / configuration (§4.7)
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    chosen_model: str
    n_agents: int = 1
    plan: str = ""
    slash_commands: list[str] = Field(default_factory=list)  # K
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    router_name: str = "heuristic_v1"
    router_score: float = 0.0


class MasterPrompt(BaseModel):
    system_preamble: str = ""
    synthesized_prompt: str = ""
    cache_control: str = "ephemeral"


class WorkflowPlan(BaseModel):
    steps: list[str] = Field(default_factory=list)
    visualizer_json: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Execution / outcomes
# ---------------------------------------------------------------------------

class ExecutionEvent(BaseModel):
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0
    wall_ms: int = 0


class Outcome(BaseModel):
    exit_code: int = 0
    diff_lines_added: int = 0
    diff_lines_removed: int = 0
    files_touched: int = 0
    tests_passed: Optional[bool] = None
    user_accepted: Optional[bool] = None
    user_thumbs: int = 0  # -1 / 0 / 1


class UserFeedback(BaseModel):
    thumbs: int = 0
    followup_prompt: Optional[str] = None
    followup_within_seconds: Optional[int] = None


# ---------------------------------------------------------------------------
# Quality scalar (§4.10)
# ---------------------------------------------------------------------------

class QualityScalar(BaseModel):
    goal_met: bool = False
    no_errors: bool = False
    matches_conventions: bool = False
    minimal_changes: bool = False

    @property
    def score(self) -> float:
        return (
            int(self.goal_met)
            + int(self.no_errors)
            + int(self.matches_conventions)
            + int(self.minimal_changes)
        ) / 4


# ---------------------------------------------------------------------------
# Session / project context
# ---------------------------------------------------------------------------

class FileRef(BaseModel):
    path: str
    content_hash: str = ""


class ProjectContext(BaseModel):
    workspace_id: str = ""
    repo_root: str = ""
    language: str = ""
    framework: str = ""
    claude_md_summary: str = ""


class FastIntent(BaseModel):
    intent: str = ""
    confidence: float = 0.0
    short_circuit: bool = False  # True → trivial/chitchat, skip pipeline


class CachedResult(BaseModel):
    trace_id: str
    master_prompt: str
    plan: str
    routing_decision: RoutingDecision
    similarity: float


# ---------------------------------------------------------------------------
# §5.3.3 / §5.3.19 — Experience Memory record (ExpeL-inspired)
# ---------------------------------------------------------------------------

class ExperienceScope(str, Enum):
    PROJECT = "project"   # scope_weight=3
    USER = "user"         # scope_weight=2
    GLOBAL = "global"     # scope_weight=1


class ExperienceRecord(BaseModel):
    """One distilled, reusable lesson with trust / staleness metadata (§5.3.19)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scope: ExperienceScope = ExperienceScope.PROJECT
    workspace_id: str = ""
    user_id: str = ""

    # Task fingerprint for retrieval scoring
    task_type: TaskType = TaskType.NEW_CODE
    tags: list[str] = Field(default_factory=list)

    # The lesson itself
    lesson: str = ""                       # one-sentence reusable insight (l_e)
    failure_modes: list[str] = Field(default_factory=list)
    successful_patterns: list[str] = Field(default_factory=list)

    # Outcome label used by retrieval bonus B(o_e)
    outcome_label: Literal["success", "failure"] = "success"

    # ExpeL operator state
    weight: int = 2                        # add starts at 2; pruned at 0
    trust_score: float = 1.0               # mutated by upvote/downvote (clamped)
    conf_count: int = 1                    # consecutive confirmations; reset on contradiction

    # Staleness metadata (§5.3.3)
    last_confirmed_run: int = 0
    last_confirmed: str = ""               # ISO timestamp
    created_at: str = ""

    archived: bool = False
