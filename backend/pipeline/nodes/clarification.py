"""Nodes 5–8 — Clarification Trigger, Clarifier, Sufficiency QA, Re-Scorer (§4.5–4.6)."""

from __future__ import annotations

import structlog
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.core.llm import llm_client
from backend.core.math_core import enrich_scores, should_clarify
from backend.core.models import (
    Clarification,
    ClarificationQuestion,
    TaskClassification,
    TaskScores,
)
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)

_MAX_CLARIFICATION_ROUNDS = 2


# ---------------------------------------------------------------------------
# Node 5: decide_clarification (pure function — §4.5)
# ---------------------------------------------------------------------------

def decide_clarification(state: OptiviaState) -> OptiviaState:
    """Conditional edge: fires clarification or passes through."""
    scores = state.get("scores")
    if scores is None:
        state["_clarify"] = False  # type: ignore[typeddict-unknown-key]
        return state

    rounds = state.get("clarification_round", 0)
    if rounds >= _MAX_CLARIFICATION_ROUNDS:
        state["_clarify"] = False  # type: ignore[typeddict-unknown-key]
        return state

    fire = should_clarify(scores, scores.confidence)
    state["_clarify"] = fire  # type: ignore[typeddict-unknown-key]
    log.info("decide_clarification", fire=fire, kappa=scores.complexity, sigma=scores.specificity)
    return state


# ---------------------------------------------------------------------------
# Node 6: generate_clarifications — Clarifier (§6.2 node 5)
# ---------------------------------------------------------------------------

class ClarifyOutput(BaseModel):
    questions: list[ClarificationQuestion]


_CLARIFIER_SYSTEM = """\
You are Optivia's Clarifier. Given an ambiguous developer prompt and its complexity scores,
produce 1–3 targeted clarifying questions that will most reduce ambiguity before execution.
Each question should target a specific scoring dimension (scope / ambiguity / risk / dependency / context_load).
Avoid obvious questions. Prefer binary or short-answer questions.
Never ask about things already specified in the prompt.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def _call_clarifier_llm(raw: str, scores_str: str) -> ClarifyOutput:
    return await llm_client.structured_generate(
        system_prompt=_CLARIFIER_SYSTEM,
        user_prompt=(
            f"Developer prompt:\n\"\"\"\n{raw}\n\"\"\"\n\n"
            f"Current scores: {scores_str}\n\n"
            "Generate 1–3 clarifying questions."
        ),
        response_model=ClarifyOutput,
    )


async def generate_clarifications(state: OptiviaState) -> OptiviaState:
    """Node: generate_clarifications — Sonnet 4.6 via instructor (EIG heuristic, Stage 1)."""
    raw = state.get("raw_prompt", "")
    scores = state.get("scores")
    scores_str = str(scores.model_dump()) if scores else "{}"

    try:
        result = await _call_clarifier_llm(raw, scores_str)
        questions = result.questions[:3]
    except Exception as exc:
        log.error("generate_clarifications.error", error=str(exc))
        questions = []

    clarification = Clarification(questions=questions)
    existing = list(state.get("clarifications", []))
    existing.append(clarification)
    state["clarifications"] = existing
    state["clarification_round"] = state.get("clarification_round", 0) + 1
    return state


# ---------------------------------------------------------------------------
# Node 7: sufficiency_qa — checks if answers are sufficient
# ---------------------------------------------------------------------------

class SufficiencyOutput(BaseModel):
    sufficient: bool
    reasoning: str


_SUFFICIENCY_SYSTEM = """\
You are Optivia's Sufficiency QA agent. Given the original prompt, the clarifying questions,
and the user's answers, determine whether the answers provide enough information to proceed
with execution without further clarification.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def _call_sufficiency_llm(prompt: str, qa_pairs: str) -> SufficiencyOutput:
    return await llm_client.structured_generate(
        system_prompt=_SUFFICIENCY_SYSTEM,
        user_prompt=f"Original prompt: {prompt}\n\nQ&A:\n{qa_pairs}",
        response_model=SufficiencyOutput,
    )


async def sufficiency_qa(state: OptiviaState) -> OptiviaState:
    """Node: sufficiency_qa — validates clarification answers."""
    clarifications = state.get("clarifications", [])
    if not clarifications:
        state["_sufficiency_passed"] = True  # type: ignore[typeddict-unknown-key]
        return state

    last = clarifications[-1]
    if not last.answers:
        # No answers yet — this node fires after user provides answers
        state["_sufficiency_passed"] = False  # type: ignore[typeddict-unknown-key]
        return state

    qa_pairs = "\n".join(
        f"Q: {q.question}\nA: {a}"
        for q, a in zip(last.questions, last.answers)
    )

    try:
        result = await _call_sufficiency_llm(state.get('raw_prompt', ''), qa_pairs)
        passed = result.sufficient
    except Exception as exc:
        log.error("sufficiency_qa.error", error=str(exc))
        passed = True  # fail open

    last.sufficiency_passed = passed
    state["_sufficiency_passed"] = passed  # type: ignore[typeddict-unknown-key]
    return state


# ---------------------------------------------------------------------------
# Node 8: re_scorer — scores (raw_prompt + answers) → v_updated (§4.6)
# ---------------------------------------------------------------------------

class ReScoreOutput(BaseModel):
    scores: TaskScores
    classification: TaskClassification


_RESCORER_SYSTEM = """\
You are Optivia's Re-Scorer. Given the original prompt plus the user's clarification answers,
re-score the five signals (scope, ambiguity, risk, dependency, context_load).
The ambiguity score should drop significantly if the user answered well.
Also update the task classification if answers changed the nature of the task.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def _call_rescorer_llm(combined: str) -> ReScoreOutput:
    return await llm_client.structured_generate(
        system_prompt=_RESCORER_SYSTEM,
        user_prompt=combined,
        response_model=ReScoreOutput,
    )


async def re_scorer(state: OptiviaState) -> OptiviaState:
    """Node: re_scorer — produces v_updated = score(raw_prompt + answers)."""
    raw = state.get("raw_prompt", "")
    clarifications = state.get("clarifications", [])

    answers_text = ""
    for c in clarifications:
        for q, a in zip(c.questions, c.answers):
            answers_text += f"Q: {q.question}\nA: {a}\n"

    combined = f"Prompt: {raw}\n\nAnswers:\n{answers_text}"

    try:
        result = await _call_rescorer_llm(combined)
        enriched = enrich_scores(result.scores)
        state["scores_updated"] = enriched
        state["task_classification"] = result.classification
    except Exception as exc:
        log.error("re_scorer.error", error=str(exc))
        # Keep original scores

    return state
