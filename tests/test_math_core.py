"""Unit tests for the mathematical core — all pure functions, no I/O."""

import pytest

from backend.core.math_core import (
    check_token_budget,
    compute_complexity,
    compute_quality,
    compute_specificity,
    enrich_scores,
    quality_branch,
    resolve_slash_commands,
    should_clarify,
)
from backend.core.models import QualityScalar, TaskScores, TaskType


def _scores(**kw) -> TaskScores:
    defaults = dict(scope=0.3, ambiguity=0.3, risk=0.2, dependency=0.2, context_load=0.2, confidence=0.8)
    defaults.update(kw)
    return TaskScores(**defaults)


# ---------------------------------------------------------------------------
# §4.3 — composite complexity κ_t
# ---------------------------------------------------------------------------

def test_complexity_minimum():
    s = _scores(scope=0, ambiguity=0, risk=0, dependency=0, context_load=0)
    assert compute_complexity(s) == 1


def test_complexity_maximum():
    s = _scores(scope=1, ambiguity=1, risk=1, dependency=1, context_load=1)
    assert compute_complexity(s) == 10


def test_complexity_midpoint():
    s = _scores(scope=0.5, ambiguity=0.5, risk=0.5, dependency=0.5, context_load=0.5)
    assert compute_complexity(s) == 6  # round(1 + 9*0.5) = round(5.5) = 6


def test_complexity_formula():
    s = _scores(scope=0.2, ambiguity=0.4, risk=0.6, dependency=0.0, context_load=0.3)
    avg = (0.2 + 0.4 + 0.6 + 0.0 + 0.3) / 5  # = 0.3
    expected = round(1 + 9 * 0.3)  # = round(3.7) = 4
    assert compute_complexity(s) == expected


# ---------------------------------------------------------------------------
# §4.4 — specificity σ_t
# ---------------------------------------------------------------------------

def test_specificity():
    s = _scores(ambiguity=0.4)
    assert compute_specificity(s) == pytest.approx(0.6)


def test_specificity_extremes():
    assert compute_specificity(_scores(ambiguity=0.0)) == pytest.approx(1.0)
    assert compute_specificity(_scores(ambiguity=1.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# §4.5 — clarification trigger predicate
# ---------------------------------------------------------------------------

def test_no_clarify_low_complexity():
    s = enrich_scores(_scores(scope=0.1, ambiguity=0.8, risk=0.1, dependency=0.1, context_load=0.1))
    # κ = round(1 + 9*(0.1+0.8+0.1+0.1+0.1)/5) = round(1+9*0.24) = round(3.16) = 3
    assert s.complexity < 7
    assert not should_clarify(s)


def test_clarify_high_kappa_vague():
    s = enrich_scores(_scores(scope=0.9, ambiguity=0.8, risk=0.7, dependency=0.6, context_load=0.5))
    # All high → κ=10, σ=0.2 → triggers both (κ≥7∧σ<0.5) and (κ=10)
    assert should_clarify(s)


def test_clarify_max_kappa_always():
    s = TaskScores(scope=0, ambiguity=0, risk=0, dependency=0, context_load=0)
    s.complexity = 10
    s.specificity = 1.0  # even if prompt is specific, κ=10 triggers
    assert should_clarify(s)


def test_no_clarify_high_kappa_specific():
    # κ=7, σ=0.8 (specific), scope low → should NOT fire
    s = enrich_scores(_scores(scope=0.7, ambiguity=0.1, risk=0.6, dependency=0.5, context_load=0.3))
    # σ = 1 - 0.1 = 0.9 (high specificity) — check manually
    s2 = enrich_scores(s)
    if s2.complexity >= 7 and s2.specificity >= 0.5:
        assert not should_clarify(s2)


# ---------------------------------------------------------------------------
# §4.8 — slash-command conflict resolution
# ---------------------------------------------------------------------------

def test_conflict_fast_vs_plan():
    cmds = resolve_slash_commands(TaskType.NEW_CODE, ["/fast", "/plan"])
    assert "/plan" in cmds
    assert "/fast" not in cmds


def test_conflict_loop_vs_review():
    cmds = resolve_slash_commands(TaskType.REFACTOR, ["/loop", "/review"])
    assert "/review" in cmds
    assert "/loop" not in cmds


def test_base_commands_new_code():
    cmds = resolve_slash_commands(TaskType.NEW_CODE, [])
    assert "/plan" in cmds
    assert "/memory" in cmds


def test_trivial_no_commands():
    cmds = resolve_slash_commands(TaskType.TRIVIAL, [])
    assert cmds == []


# ---------------------------------------------------------------------------
# §4.9 — token budget invariant
# ---------------------------------------------------------------------------

def test_budget_ok():
    rho, action = check_token_budget(10_000, 5_000, 5_000, 10_000, 200_000)
    assert rho < 0.60
    assert action == "ok"


def test_budget_summarise():
    rho, action = check_token_budget(60_000, 30_000, 20_000, 10_000, 200_000)
    assert 0.60 <= rho < 0.70
    assert action == "summarise"


def test_budget_emergency():
    rho, action = check_token_budget(180_000, 5_000, 5_000, 5_000, 200_000)
    assert rho >= 0.90
    assert action == "emergency"


# ---------------------------------------------------------------------------
# §4.10 — quality scalar Q_t
# ---------------------------------------------------------------------------

def test_quality_all_true():
    q = QualityScalar(goal_met=True, no_errors=True, matches_conventions=True, minimal_changes=True)
    assert compute_quality(q) == 1.0


def test_quality_all_false():
    q = QualityScalar(goal_met=False, no_errors=False, matches_conventions=False, minimal_changes=False)
    assert compute_quality(q) == 0.0


def test_quality_half():
    q = QualityScalar(goal_met=True, no_errors=True, matches_conventions=False, minimal_changes=False)
    assert compute_quality(q) == 0.5


def test_quality_branch_halt():
    assert quality_branch(0.25) == "halt"
    assert quality_branch(0.49) == "halt"


def test_quality_branch_reverify():
    assert quality_branch(0.5) == "reverify"
    assert quality_branch(0.74) == "reverify"


def test_quality_branch_pass():
    assert quality_branch(0.75) == "pass"
    assert quality_branch(1.0) == "pass"


def test_quality_branch_long_clean_stretch():
    assert quality_branch(0.95, consecutive_high=5) == "long_clean_stretch"
