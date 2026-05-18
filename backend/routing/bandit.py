"""
Agent 10B — Discounted Contextual Bandit Selector (§5.3.12).

Implements D-LinUCB:
    a* = argmax_{a ∈ A_allowed} ( θ̂_a · x_t + α · sqrt(x_tᵀ · A_a⁻¹ · x_t) )
    where θ̂_a = A_a⁻¹ · b_a

Update rule (per chosen arm a*):
    A_a* ← γ · A_a* + x_t x_tᵀ
    b_a* ← γ · b_a* + r_t · x_t

Reward:
    r_t = clamp(Q_t + b_first - λ_c·ĉ - λ_l·l̂ - λ_R·R - λ_T·T̂, 0, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from backend.core.models import TaskScores, TaskType
from backend.routing.safety_router import Arm

log = structlog.get_logger(__name__)

# Hyper-parameters (§5.3.12)
GAMMA = 0.97
ALPHA = 1.0          # exploration coefficient
RIDGE = 1.0          # initial λI ridge on A_a

# Reward weights
LAMBDA_COST = 0.20
LAMBDA_LATENCY = 0.10
LAMBDA_RETRY = 0.15
LAMBDA_TOKEN = 0.10
BONUS_FIRST_TRY = 0.10
COLD_START_N = 10    # below this, fall back to heuristic

# Context vector dimension — must stay stable across reloads
TASK_TYPES = [
    TaskType.NEW_CODE, TaskType.DEBUG, TaskType.REFACTOR,
    TaskType.REVIEW, TaskType.EXPLAIN, TaskType.LONG,
    TaskType.TRIVIAL, TaskType.META,
]
EXECUTORS = ["claude_code", "remote_api", "ide"]
# κ/10 + 8 one-hot task + 5 signals + ρ + avg_q + retry + 3 one-hot executor + ctx_size_norm + cpl_norm
CONTEXT_DIM = 1 + len(TASK_TYPES) + 5 + 1 + 1 + 1 + len(EXECUTORS) + 1 + 1


def build_context_vector(
    *,
    scores: TaskScores,
    task_type: TaskType,
    budget_rho: float,
    avg_quality: float,
    retry_rate: float,
    executor: str = "claude_code",
    context_size_norm: float = 0.0,
    cpl_norm: float = 0.0,
) -> np.ndarray:
    x = np.zeros(CONTEXT_DIM, dtype=np.float64)
    i = 0
    x[i] = scores.complexity / 10.0; i += 1
    for tt in TASK_TYPES:
        x[i] = 1.0 if tt == task_type else 0.0; i += 1
    x[i] = scores.scope; i += 1
    x[i] = scores.ambiguity; i += 1
    x[i] = scores.risk; i += 1
    x[i] = scores.dependency; i += 1
    x[i] = scores.context_load; i += 1
    x[i] = min(1.0, max(0.0, budget_rho)); i += 1
    x[i] = min(1.0, max(0.0, avg_quality)); i += 1
    x[i] = min(1.0, max(0.0, retry_rate)); i += 1
    for ex in EXECUTORS:
        x[i] = 1.0 if executor == ex else 0.0; i += 1
    x[i] = min(1.0, max(0.0, context_size_norm)); i += 1
    x[i] = min(1.0, max(0.0, cpl_norm)); i += 1
    return x


def compute_reward(
    *,
    quality: float,
    monetary_cost_norm: float,
    latency_norm: float,
    retries: int,
    token_waste_norm: float,
    first_try_success: bool,
) -> float:
    b_first = BONUS_FIRST_TRY if (first_try_success and quality >= 0.75) else 0.0
    r = (
        quality
        + b_first
        - LAMBDA_COST * monetary_cost_norm
        - LAMBDA_LATENCY * latency_norm
        - LAMBDA_RETRY * (retries / max(1, retries + 1))
        - LAMBDA_TOKEN * token_waste_norm
    )
    return max(0.0, min(1.0, r))


class BanditPosterior:
    """Per-arm A_a (d×d) and b_a (d) with discounted updates."""

    def __init__(self, d: int):
        self.d = d
        self.A = RIDGE * np.eye(d)
        self.b = np.zeros(d)
        self.n_obs = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any], d: int) -> "BanditPosterior":
        p = cls(d)
        try:
            a_raw = raw.get("a_matrix") or []
            b_raw = raw.get("b_vector") or []
            if a_raw and b_raw and len(a_raw) == d:
                p.A = np.array(a_raw, dtype=np.float64)
                p.b = np.array(b_raw, dtype=np.float64)
                p.n_obs = int(raw.get("n_observations", 0))
        except Exception as exc:
            log.warning("bandit.posterior_load_error", error=str(exc))
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_matrix": self.A.tolist(),
            "b_vector": self.b.tolist(),
            "n_observations": self.n_obs,
        }

    def ucb_score(self, x: np.ndarray, alpha: float = ALPHA) -> float:
        try:
            A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(self.A)
        theta = A_inv @ self.b
        mean = float(theta @ x)
        bonus = float(alpha * np.sqrt(max(0.0, x @ A_inv @ x)))
        return mean + bonus

    def update(self, x: np.ndarray, reward: float, gamma: float = GAMMA) -> None:
        self.A = gamma * self.A + np.outer(x, x)
        self.b = gamma * self.b + reward * x
        self.n_obs += 1


class BanditEnsemble:
    """Holds posteriors keyed by arm_id. Loads/saves through db_client."""

    def __init__(self, posteriors: dict[str, BanditPosterior]):
        self.posteriors = posteriors

    @classmethod
    async def load(cls, workspace_id: str) -> "BanditEnsemble":
        posteriors: dict[str, BanditPosterior] = {}
        try:
            from backend.db.client import db_client
            raw = await db_client.load_bandit_state(workspace_id)
            for arm_id, payload in raw.items():
                posteriors[arm_id] = BanditPosterior.from_dict(payload, CONTEXT_DIM)
        except Exception as exc:
            log.warning("bandit.load_error", error=str(exc))
        return cls(posteriors)

    def get(self, arm_id: str) -> BanditPosterior:
        if arm_id not in self.posteriors:
            self.posteriors[arm_id] = BanditPosterior(CONTEXT_DIM)
        return self.posteriors[arm_id]

    def select(
        self,
        x: np.ndarray,
        allowed: list[Arm],
        *,
        cold_start_fallback: Arm | None = None,
    ) -> tuple[Arm, float, dict[str, float]]:
        """Returns (chosen_arm, ucb_score, all_arm_scores). Cold-start → fallback."""
        total_obs = sum(self.get(a.arm_id).n_obs for a in allowed)
        if total_obs < COLD_START_N and cold_start_fallback is not None:
            scores = {a.arm_id: 0.0 for a in allowed}
            return cold_start_fallback, 0.0, scores

        scores: dict[str, float] = {}
        for arm in allowed:
            scores[arm.arm_id] = self.get(arm.arm_id).ucb_score(x)
        best = max(allowed, key=lambda a: scores[a.arm_id])
        return best, scores[best.arm_id], scores

    async def save_arm(self, workspace_id: str, arm_id: str) -> None:
        p = self.get(arm_id)
        try:
            from backend.db.client import db_client
            await db_client.save_bandit_arm(
                workspace_id=workspace_id,
                arm_id=arm_id,
                d=p.d,
                a_matrix=p.A.tolist(),
                b_vector=p.b.tolist(),
                n_observations=p.n_obs,
            )
        except Exception as exc:
            log.warning("bandit.save_error", error=str(exc))
