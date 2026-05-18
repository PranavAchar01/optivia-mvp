"""Routing module — Protocol + three Stage 1 implementations + D-LinUCB bandit (§4.4, §5.3.12)."""

from backend.routing.base import Router, RoutingContext
from backend.routing.heuristic import HeuristicRouter
from backend.routing.routellm import RouteLLMRouter
from backend.routing.llm_judge import LLMJudgeRouter
from backend.routing.shadow import ShadowRouterRunner, get_router_runner
from backend.routing.safety_router import Arm, arm_catalog, get_allowed_arms
from backend.routing.bandit import (
    BanditEnsemble,
    BanditPosterior,
    build_context_vector,
    compute_reward,
    CONTEXT_DIM,
)

__all__ = [
    "Router",
    "RoutingContext",
    "HeuristicRouter",
    "RouteLLMRouter",
    "LLMJudgeRouter",
    "ShadowRouterRunner",
    "get_router_runner",
    # Agent 10A
    "Arm",
    "arm_catalog",
    "get_allowed_arms",
    # Agent 10B
    "BanditEnsemble",
    "BanditPosterior",
    "build_context_vector",
    "compute_reward",
    "CONTEXT_DIM",
]
