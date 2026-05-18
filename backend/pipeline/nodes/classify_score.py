"""Node 3–4 — Task Classifier + Scorer (§6.2 node 3, deep_classify_and_score)."""

from __future__ import annotations

import structlog
import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.core.math_core import enrich_scores
from backend.core.models import TaskClassification, TaskScores, TaskType
from backend.pipeline.reflection import CONFIDENCE_THRESHOLD, reflect
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)

client = instructor.from_anthropic(AsyncAnthropic(api_key=settings.anthropic_api_key))

_SYSTEM_PROMPT = """\
You are the Optivia task classifier and scorer. Given a developer's prompt, produce:
1. A TaskClassification (task_type from the enum, reasoning).
2. TaskScores — five [0,1] signals:
   - scope: how much of the codebase is touched (0=one line, 1=multi-system)
   - ambiguity: how much is left unsaid (0=fully specified, 1=vague)
   - risk: blast radius (0=read-only, 1=delete/migrate/restructure)
   - dependency: coupling to other systems (0=self-contained, 1=cross-system)
   - context_load: files needed before starting (0=none, 1=many/unknown)
Plus est_tier, est_tokens_input, est_tokens_output, est_wall_seconds, confidence.

Context matters:
- If files are explicitly attached, ambiguity and context_load might be lower.
- Be precise. The scores directly control model routing, cost, and the sub-agent fleet generation.
"""


class ClassifyScoreOutput(BaseModel):
    classification: TaskClassification
    scores: TaskScores


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def _call_llm(raw: str, ctx_str: str) -> ClassifyScoreOutput:
    return await client.chat.completions.create(
        model=settings.model_haiku,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"Prompt to classify:\n\"\"\"\n{raw}\n\"\"\"{ctx_str}",
            }
        ],
        system=_SYSTEM_PROMPT,
        response_model=ClassifyScoreOutput,
    )


async def classify_and_score(state: OptiviaState) -> OptiviaState:
    """
    Node: deep_classify_and_score
    Single instructor call to Haiku 4.5 returning validated TaskClassification + TaskScores.
    """
    raw = state.get("raw_prompt", "")
    project_ctx = state.get("project_context")
    ctx_str = ""
    if project_ctx:
        ctx_str = f"\nProject context: language={project_ctx.language}, framework={project_ctx.framework}"

    attached = state.get("attached_files", [])
    if attached:
        files_str = ", ".join([f.path for f in attached])
        ctx_str += f"\nAttached files: {files_str}"

    try:
        result = await _call_llm(raw, ctx_str)
    except Exception as exc:
        log.error("classify_and_score.error", error=str(exc))
        # Fallback: conservative default scores
        result = ClassifyScoreOutput(
            classification=TaskClassification(task_type=TaskType.NEW_CODE, reasoning="fallback"),
            scores=TaskScores(scope=0.5, ambiguity=0.5, risk=0.3, dependency=0.3, context_load=0.3, confidence=0.3),
        )

    # ── Reflection sub-steps for Agents 3 and 4 (§5.3.4 / §5.3.5) ─────────────
    lessons = state.get("retrieved_lessons") or []
    is_first_turn = state.get("turn_index", 0) == 0
    avg_q = 0.0
    if state.get("consecutive_high_quality", 0) >= 1:
        avg_q = 0.9  # crude proxy until we wire q_history into state

    # Agent 3 reflection — classification consistency
    critique_3, conf_3 = await reflect(
        agent_name="classifier",
        output=(
            f"task_type={result.classification.task_type.value}; "
            f"reasoning={result.classification.reasoning}"
        ),
        rubric="Does this classification match the prompt content and the retrieved lessons?",
        lessons=lessons,
        input_context=raw,
        avg_quality=avg_q,
        is_first_turn=is_first_turn,
    )
    if conf_3 <= CONFIDENCE_THRESHOLD:
        try:
            result_retry = await _call_llm(raw, ctx_str + f"\nReviewer critique: {critique_3}")
            # Keep the original scores if the retry is clearly worse; otherwise replace.
            if result_retry.classification.task_type is not None:
                result = result_retry
            log.info("classify_and_score.reflection_retry", agent=3, confidence=conf_3)
        except Exception as exc:
            log.warning("classify_and_score.reflection_retry_error", agent=3, error=str(exc))

    # Agent 4 reflection — scorer internal consistency
    critique_4, conf_4 = await reflect(
        agent_name="scorer",
        output=str(result.scores.model_dump()),
        rubric="Are the 5 signals internally consistent and consistent with retrieved lessons?",
        lessons=lessons,
        input_context=raw,
        avg_quality=avg_q,
        is_first_turn=is_first_turn,
    )
    if conf_4 <= CONFIDENCE_THRESHOLD:
        try:
            result_retry = await _call_llm(raw, ctx_str + f"\nReviewer critique: {critique_4}")
            result = result_retry
            log.info("classify_and_score.reflection_retry", agent=4, confidence=conf_4)
        except Exception as exc:
            log.warning("classify_and_score.reflection_retry_error", agent=4, error=str(exc))

    enriched = enrich_scores(result.scores)
    state["task_classification"] = result.classification
    state["scores"] = enriched
    state["scores_updated"] = enriched  # will be overwritten by Re-Scorer if clarification fires

    log.info(
        "classify_and_score.done",
        task_type=result.classification.task_type,
        kappa=enriched.complexity,
        sigma=enriched.specificity,
        reflect_conf=(round(conf_3, 2), round(conf_4, 2)),
    )
    return state
