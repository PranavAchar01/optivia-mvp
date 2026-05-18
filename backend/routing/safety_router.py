"""
Agent 10A — Deterministic Safety Router (§5.3.11).

Defines the 6 named candidate arms A0..A5 and prunes them with deterministic
risk / budget / policy rules. The surviving set A_allowed is consumed by the
D-LinUCB selector (Agent 10B).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import settings
from backend.core.models import TaskType


@dataclass(frozen=True)
class Arm:
    arm_id: str             # A0..A5
    model_tier: str         # "haiku_fastest" | "haiku_fast" | "sonnet" | "opus"
    model: str              # concrete model identifier
    n_agents: int           # base fleet size for this arm
    planning_on: bool
    verification: str       # "none" | "on_error" | "on_failure" | "light" | "full" | "always"
    expected_cost_ceiling: float  # USD upper bound per turn for budget gating
    speed_rank: int         # 0 = fastest


def arm_catalog() -> list[Arm]:
    return [
        Arm("A0", "haiku_fastest", settings.model_haiku, 1, False, "none",       0.02, 0),
        Arm("A1", "haiku_fast",    settings.model_haiku, 1, False, "on_error",   0.04, 1),
        Arm("A2", "balanced",      settings.model_haiku, 2, False, "on_failure", 0.10, 2),
        Arm("A3", "balanced",      settings.model_sonnet, 2, True, "light",      0.25, 3),
        Arm("A4", "strong",        settings.model_sonnet, 3, True, "full",       0.50, 4),
        Arm("A5", "strongest",     settings.model_opus,   5, True, "always",     1.20, 5),
    ]


def get_allowed_arms(
    *,
    risk: float,
    complexity: int,
    task_type: TaskType,
    budget_rho: float,
    remaining_budget_usd: float | None = None,
    policy_require_verifier: bool = False,
) -> list[Arm]:
    """
    Applies the rejection rules from §5.3.11:
      - δ_r > 0.75            → drop A0/A1
      - debug/refactor + κ>8  → drop planning-off arms (A0..A2)
      - ρ_t > 0.90            → drop arms whose ceiling exceeds remaining budget
      - policy requires verifier → drop arms with no/on_error verification
    """
    arms = arm_catalog()

    def keep(a: Arm) -> bool:
        if risk > 0.75 and a.arm_id in {"A0", "A1"}:
            return False
        if task_type in {TaskType.DEBUG, TaskType.REFACTOR} and complexity > 8 and not a.planning_on:
            return False
        if budget_rho > 0.90:
            if remaining_budget_usd is not None and a.expected_cost_ceiling > remaining_budget_usd:
                return False
        if policy_require_verifier and a.verification in {"none", "on_error"}:
            return False
        return True

    allowed = [a for a in arms if keep(a)]
    if not allowed:
        # Always leave at least the cheapest safe fallback so the bandit has a choice
        allowed = [arms[2]]  # A2 — balanced, on_failure verification
    return allowed
