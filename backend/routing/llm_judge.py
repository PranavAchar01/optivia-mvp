"""
LLMJudgeRouter — Haiku call with structured output that picks a model (§4.4).

This router is mostly useful as a tiebreaker and as a generator of Stage 2
training labels (RouteLLM's data-augmentation trick — §5.4 head 1).
"""

from __future__ import annotations

import instructor
import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from backend.config import settings
from backend.core.math_core import resolve_slash_commands
from backend.core.models import RoutingDecision
from backend.routing.base import Router, RoutingContext

log = structlog.get_logger(__name__)

_client = instructor.from_anthropic(AsyncAnthropic(api_key=settings.anthropic_api_key))


class JudgeOutput(BaseModel):
    chosen_model: str = Field(description="One of haiku, sonnet, opus")
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    suggested_n_agents: int = Field(ge=1, le=12)


_SYSTEM = """\
You are Optivia's LLM-judge router. Given a developer prompt with its
classification and scoring vector, pick the most cost-effective Anthropic
model from {haiku, sonnet, opus}.

Guidelines:
- haiku is correct for trivial edits, single-file changes, format/rename
- sonnet is the default for most coding work
- opus is reserved for genuine architectural reasoning (κ ≥ 8, multi-system)

Prefer the cheapest model that you believe will succeed. Include a brief
reasoning string and your confidence in the decision.
"""


_KEY_TO_MODEL: dict[str, str] = {
    "haiku":  settings.model_haiku,
    "sonnet": settings.model_sonnet,
    "opus":   settings.model_opus,
}


class LLMJudgeRouter:
    name = "llm_judge_haiku_v1"

    async def route(self, ctx: RoutingContext) -> RoutingDecision:
        s = ctx.scores
        user_msg = (
            f"Prompt:\n\"\"\"\n{ctx.raw_prompt[:1000]}\n\"\"\"\n\n"
            f"Task type: {ctx.task_classification.task_type.value}\n"
            f"Scores: κ={s.complexity}, σ={s.specificity:.2f}, "
            f"risk={s.risk:.2f}, scope={s.scope:.2f}, "
            f"dependency={s.dependency:.2f}, context_load={s.context_load:.2f}"
        )

        try:
            result: JudgeOutput = await _client.chat.completions.create(
                model=settings.model_haiku,
                max_tokens=256,
                messages=[{"role": "user", "content": user_msg}],
                system=_SYSTEM,
                response_model=JudgeOutput,
            )
        except Exception as exc:
            log.warning("llm_judge.error", error=str(exc))
            return RoutingDecision(
                chosen_model=settings.model_sonnet,
                n_agents=1,
                router_name=self.name,
                router_score=0.0,
            )

        key = result.chosen_model.lower().strip()
        if key not in _KEY_TO_MODEL:
            key = "sonnet"
        chosen = _KEY_TO_MODEL[key]
        commands = resolve_slash_commands(ctx.task_classification.task_type, [])

        return RoutingDecision(
            chosen_model=chosen,
            n_agents=result.suggested_n_agents,
            plan=result.reasoning[:200],
            slash_commands=commands,
            router_name=self.name,
            router_score=float(result.confidence),
        )
