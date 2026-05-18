"""
Mathematical core — §4 of the unified architecture doc.

All functions here are pure (no I/O) so they are trivially testable.
"""

from __future__ import annotations

from backend.core.models import (
    ClarificationQuestion,
    QualityScalar,
    RoutingDecision,
    TaskScores,
    TaskType,
)
from backend.config import settings


# ---------------------------------------------------------------------------
# §4.2 / §4.3 — Composite complexity κ_t
# ---------------------------------------------------------------------------

def compute_complexity(scores: TaskScores, weights: dict[str, float] | None = None) -> int:
    """κ_t = round(1 + 9 · (δ_s + δ_a + δ_r + δ_d + δ_c) / 5) ∈ {1,…,10}"""
    if weights is None:
        # Default equal weights (w_i = 1/5 per §4.3)
        avg = (scores.scope + scores.ambiguity + scores.risk + scores.dependency + scores.context_load) / 5
    else:
        avg = sum(weights.get(k, 0.2) * v for k, v in {
            "scope": scores.scope,
            "ambiguity": scores.ambiguity,
            "risk": scores.risk,
            "dependency": scores.dependency,
            "context_load": scores.context_load,
        }.items())
    return round(1 + 9 * avg)


# §4.4 — Specificity σ_t
def compute_specificity(scores: TaskScores) -> float:
    """σ_t = 1 - δ_a"""
    return 1.0 - scores.ambiguity


def enrich_scores(raw: TaskScores) -> TaskScores:
    """Fill computed fields (κ_t, σ_t) from the five raw signals."""
    raw.complexity = compute_complexity(raw)
    raw.specificity = compute_specificity(raw)
    return raw


# ---------------------------------------------------------------------------
# §4.5 — Clarification trigger predicate
# ---------------------------------------------------------------------------

def should_clarify(
    scores: TaskScores,
    classifier_confidence: float | None = None,
) -> bool:
    """
    ask_questions(κ_t, σ_t, δ_s, c_t) =
        (κ_t ≥ 7 ∧ σ_t < 0.5) ∨ (δ_s ≥ 0.7 ∧ c_t < 0.5) ∨ (κ_t = 10)

    Uses per-tenant configurable thresholds from settings.
    """
    kappa = scores.complexity
    sigma = scores.specificity
    delta_s = scores.scope
    c_t = classifier_confidence if classifier_confidence is not None else scores.confidence

    high_complexity_vague = (
        kappa >= settings.clarify_kappa_threshold
        and sigma < settings.clarify_sigma_threshold
    )
    high_scope_low_confidence = (
        delta_s >= settings.clarify_scope_threshold
        and c_t < settings.clarify_confidence_threshold
    )
    max_complexity = kappa == 10

    return high_complexity_vague or high_scope_low_confidence or max_complexity


# ---------------------------------------------------------------------------
# §4.7 — Configuration function (Stage 1 lookup table)
# ---------------------------------------------------------------------------

# Stage-1 lookup table — κ → (model_tier_label, planning_on, n_agents_base, ask_questions)
_ROUTING_TABLE: dict[int, tuple[str, bool, int, bool]] = {
    1: ("haiku_fastest", False, 1, False),
    2: ("haiku_fastest", False, 1, False),
    3: ("haiku_fastest", False, 1, False),
    4: ("haiku_fast", False, 1, False),
    5: ("haiku_fast", False, 1, False),
    6: ("sonnet_balanced", False, 1, False),
    7: ("sonnet_balanced", False, 2, True),
    8: ("sonnet_strong", True, 2, True),
    9: ("opus_strong", True, 3, True),
    10: ("opus_verifier", True, 3, True),
}

# n_agents super-linear scaling per spec §5.3.5 — κ→N lookup
_N_AGENTS_NONLINEAR: dict[int, int] = {
    1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 5, 7: 8, 8: 12, 9: 20, 10: 35,
}

def _model_for_tier(tier_label: str) -> str:
    mapping = {
        "haiku_fastest": settings.model_haiku,
        "haiku_fast": settings.model_haiku,
        "sonnet_balanced": settings.model_sonnet,
        "sonnet_strong": settings.model_sonnet,
        "opus_strong": settings.model_opus,
        "opus_verifier": settings.model_opus,
    }
    return mapping.get(tier_label, settings.model_sonnet)


def make_routing_decision(
    scores: TaskScores,
    task_type: TaskType,
    slash_commands: list[str],
) -> RoutingDecision:
    """Pure configuration function C_t = f(v_updated, I) (§4.7)."""
    kappa = scores.complexity
    tier_label, planning_on, n_agents_base, _ = _ROUTING_TABLE.get(kappa, _ROUTING_TABLE[5])
    model = _model_for_tier(tier_label)
    n_agents = _N_AGENTS_NONLINEAR.get(kappa, n_agents_base)

    resolved_commands = resolve_slash_commands(task_type, slash_commands)

    return RoutingDecision(
        chosen_model=model,
        n_agents=n_agents,
        plan="" if not planning_on else "planning_enabled",
        slash_commands=resolved_commands,
        router_name="heuristic_v1",
        router_score=float(kappa) / 10,
    )


# ---------------------------------------------------------------------------
# §4.8 — Slash-command routing with conflict resolution
# ---------------------------------------------------------------------------

_BASE_COMMANDS: dict[TaskType, list[str]] = {
    TaskType.NEW_CODE:  ["/plan", "/memory"],
    TaskType.DEBUG:     ["/debug", "/memory", "/rewind"],
    TaskType.REFACTOR:  ["/plan", "/batch", "/review", "/rewind"],
    TaskType.LONG:      ["/plan", "/compact", "/loop", "/rewind"],
    TaskType.REVIEW:    ["/review", "/debug"],
    TaskType.EXPLAIN:   ["/review"],
    TaskType.TRIVIAL:   [],
    TaskType.META:      [],
}

# Conflict matrix: (cmd_a, cmd_b) → winner
_CONFLICT_WINNER: dict[tuple[str, str], str] = {
    ("/fast", "/plan"): "/plan",
    ("/loop", "/review"): "/review",
    ("/memory", "/clear"): "/memory",
    ("/fast", "/debug"): "/debug",
}

def resolve_slash_commands(task_type: TaskType, candidates: list[str]) -> list[str]:
    """K = base_set(τ) ∩ ¬conflicts(K_candidate)  (§4.8)"""
    base = set(_BASE_COMMANDS.get(task_type, []))
    all_cmds = list(base | set(candidates))

    # Remove losers from conflict pairs
    losers: set[str] = set()
    for (a, b), winner in _CONFLICT_WINNER.items():
        if a in all_cmds and b in all_cmds:
            loser = b if winner == a else a
            losers.add(loser)

    return sorted(c for c in all_cmds if c not in losers)


# ---------------------------------------------------------------------------
# §4.9 — Token-budget invariant
# ---------------------------------------------------------------------------

def check_token_budget(
    obs_tokens: int,
    memory_tokens: int,
    plan_tokens: int,
    action_tokens: int,
    budget: int,
) -> tuple[float, str]:
    """
    Returns (ρ, action) where ρ = Σ/W.
    action is one of: "ok", "summarise", "drop", "switch_model", "emergency".
    """
    total = obs_tokens + memory_tokens + plan_tokens + action_tokens
    rho = total / budget if budget > 0 else 1.0

    if rho >= 0.90:
        action = "emergency"
    elif rho >= 0.80:
        action = "switch_model"
    elif rho >= 0.70:
        action = "drop"
    elif rho >= 0.60:
        action = "summarise"
    else:
        action = "ok"

    return rho, action


# ---------------------------------------------------------------------------
# §4.10 — Quality scalar Q_t
# ---------------------------------------------------------------------------

def compute_quality(q: QualityScalar) -> float:
    """Q_t = ¼(1_goal + 1_no_err + 1_conv + 1_minimal) ∈ {0, 0.25, 0.5, 0.75, 1}"""
    return q.score


def quality_branch(q_score: float, consecutive_high: int = 0) -> str:
    """
    Returns routing decision:
      "halt"      — Q_t < 0.5 → force replan via loop back to Scorer
      "reverify"  — Q_t < 0.75 → spawn verifier inside executor
      "long_clean_stretch" — N consecutive Q_t > 0.9 → relax overhead
      "pass"      — otherwise
    """
    if q_score < 0.5:
        return "halt"
    if q_score < 0.75:
        return "reverify"
    if consecutive_high >= settings.quality_long_clean_n:
        return "long_clean_stretch"
    return "pass"
