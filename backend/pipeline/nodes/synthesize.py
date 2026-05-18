"""Node 9 — Prompt Synthesizer (§6.2 node 6): DSPy ChainOfThought + Anthropic caching."""

from __future__ import annotations

import structlog

from backend.core.models import MasterPrompt
from backend.pipeline.reflection import CONFIDENCE_THRESHOLD, reflect
from backend.pipeline.state import OptiviaState
from backend.synthesis import SYSTEM_PREAMBLE, synthesize

log = structlog.get_logger(__name__)


async def synthesize_master_prompt(state: OptiviaState) -> OptiviaState:
    """
    Node: synthesize_master_prompt — DSPy-shaped signature, Anthropic-backed
    implementation with prompt caching on the system preamble (§4.8, §8.1).
    """
    raw = state.get("raw_prompt", "")
    task_cls = state.get("task_classification")
    scores_updated = state.get("scores_updated") or state.get("scores")
    clarifications = state.get("clarifications", [])
    project_ctx = state.get("project_context")

    # ── Optional Stage-2 hook: retrieve top-3 similar successful exemplars ────
    exemplar_block = ""
    embedding = state.get("_prompt_embedding")  # type: ignore[typeddict-item]
    workspace_id = state.get("workspace_id", "")
    if embedding and any(embedding) and workspace_id:
        try:
            from backend.db.client import db_client
            exemplars = await db_client.similar_exemplars(
                embedding=embedding,
                workspace_id=workspace_id,
                task_type=task_cls.task_type.value if task_cls else None,
                k=3,
            )
            if exemplars:
                exemplar_block = "\n\nSimilar successful prompts in this workspace:\n"
                for i, ex in enumerate(exemplars, 1):
                    exemplar_block += (
                        f"\n[{i}] sim={ex.get('similarity', 0):.2f} — "
                        f"\"{(ex.get('raw_prompt') or '')[:100]}\""
                    )
        except Exception as exc:
            log.warning("synthesize.exemplar_lookup_error", error=str(exc))

    # ── Compose DSPy-shaped inputs ────────────────────────────────────────────
    task_type = task_cls.task_type.value if task_cls else "unknown"

    if scores_updated:
        scores_summary = (
            f"κ={scores_updated.complexity}/10 σ={scores_updated.specificity:.2f} "
            f"risk={scores_updated.risk:.2f} scope={scores_updated.scope:.2f} "
            f"dep={scores_updated.dependency:.2f}"
        )
    else:
        scores_summary = "(unknown)"

    qa_lines = []
    for c in clarifications:
        for q, a in zip(c.questions, c.answers):
            qa_lines.append(f"Q: {q.question}\nA: {a}")
    clarifications_str = "\n".join(qa_lines) if qa_lines else ""

    if project_ctx:
        project_str = f"language={project_ctx.language}, framework={project_ctx.framework}"
        if project_ctx.claude_md_summary:
            project_str += f"\nCLAUDE.md:\n{project_ctx.claude_md_summary[:500]}"
    else:
        project_str = ""

    # Inject exemplars as part of the raw_prompt context window
    raw_with_exemplars = raw + exemplar_block

    synthesized, in_tokens, out_tokens = await synthesize(
        raw_prompt=raw_with_exemplars,
        task_type=task_type,
        scores_summary=scores_summary,
        clarifications=clarifications_str,
        project_context=project_str,
    )

    state["obs_tokens"] = state.get("obs_tokens", 0) + in_tokens
    state["action_tokens"] = state.get("action_tokens", 0) + out_tokens

    # ── Reflection sub-step for Agent 9 (§5.3.10) ─────────────────────────────
    lessons = state.get("retrieved_lessons") or []
    is_first_turn = state.get("turn_index", 0) == 0
    avg_q = 0.9 if state.get("consecutive_high_quality", 0) >= 1 else 0.0

    critique, confidence = await reflect(
        agent_name="synthesizer",
        output=synthesized,
        rubric=(
            "Does the synthesized master prompt embed the retrieved lessons as task "
            "constraints, and does it preserve the user's original intent?"
        ),
        lessons=lessons,
        input_context=raw,
        avg_quality=avg_q,
        is_first_turn=is_first_turn,
    )

    if confidence <= CONFIDENCE_THRESHOLD:
        try:
            retry_text, in2, out2 = await synthesize(
                raw_prompt=raw_with_exemplars + f"\n\n[Reviewer critique]: {critique}",
                task_type=task_type,
                scores_summary=scores_summary,
                clarifications=clarifications_str,
                project_context=project_str,
            )
            synthesized = retry_text
            state["obs_tokens"] = state.get("obs_tokens", 0) + in2
            state["action_tokens"] = state.get("action_tokens", 0) + out2
            log.info("synthesize.reflection_retry", confidence=confidence)
        except Exception as exc:
            log.warning("synthesize.reflection_retry_error", error=str(exc))

    state["master_prompt"] = MasterPrompt(
        system_preamble=SYSTEM_PREAMBLE,
        synthesized_prompt=synthesized,
        cache_control="ephemeral",
    )

    log.info(
        "synthesize_master_prompt.done",
        prompt_len=len(synthesized),
        reflect_conf=round(confidence, 2),
    )
    return state
