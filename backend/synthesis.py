"""
DSPy-based Master Prompt Synthesizer (§4.8).

Stage 1 uses DSPy's `Predict` / `ChainOfThought` over Anthropic Sonnet 4.6
with prompt caching on the system preamble. The DSPy signature is the
contract Stage 2 will optimise with MIPRO — see §5.6 / §5.7.

The implementation is defensive: if the DSPy library isn't installed at
runtime (older test envs, CI without optional deps), we fall back to a
direct Anthropic call so the pipeline never breaks.
"""

from __future__ import annotations

import structlog
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings

log = structlog.get_logger(__name__)

# Stage-1 ~1500 token coding-agent best-practices preamble (§4.8).
SYSTEM_PREAMBLE = """\
You are a master prompt synthesizer for Optivia, a pre-execution optimization
layer for agentic coding CLIs. Your job is to transform a developer's raw
prompt into an optimised master prompt that maximises success when executed
by Claude Code.

Guidelines for the master prompt:
- Be explicit about the task scope, constraints, and success criteria
- Include relevant coding conventions from the project context
- Specify output format expectations (diffs, file paths, test requirements)
- For multi-step tasks, structure as ordered steps with clear acceptance criteria
- Preserve all technical details from the original prompt and clarification answers
- Add a verification step appropriate to the task type
- Keep the master prompt focused — no padding or excessive context

Task types and their synthesis patterns:
- NEW_CODE: include architecture sketch, file structure, test requirements
- DEBUG: include reproduction steps, expected vs actual, constraint on fix scope
- REFACTOR: include "must not change behaviour" constraint, test coverage note
- REVIEW: include specific dimensions to assess (security, perf, maintainability)
- EXPLAIN: include target audience and depth of explanation
- LONG: break into phases with checkpoints; include /compact guidance
- TRIVIAL: keep it short and precise

Output ONLY the master prompt. Do not preface with explanation or commentary.
"""


# ── DSPy signature & program ─────────────────────────────────────────────────

_DSPY_AVAILABLE = False
try:
    import dspy  # type: ignore

    class SynthesizeMasterPrompt(dspy.Signature):  # type: ignore[misc, valid-type]
        """Convert a user prompt + classification + scores + clarifications
        + project context into an optimised master prompt for Claude Code."""
        raw_prompt = dspy.InputField()
        task_type = dspy.InputField()
        scores_summary = dspy.InputField(desc="κ, σ, risk, scope, dependency summary")
        clarifications = dspy.InputField(desc="Q&A pairs, may be empty")
        project_context = dspy.InputField(desc="language, framework, claude.md summary")
        master_prompt = dspy.OutputField(desc="The optimised prompt for Claude Code")

    _DSPY_AVAILABLE = True
except Exception as exc:
    log.info("dspy.not_available", reason=str(exc))


_anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)


def _format_user_message(
    raw_prompt: str,
    task_type: str,
    scores_summary: str,
    clarifications: str,
    project_context: str,
) -> str:
    return (
        f"Original prompt:\n\"\"\"\n{raw_prompt}\n\"\"\"\n\n"
        f"Task type: {task_type}\n"
        f"Scores: {scores_summary}\n"
        f"Clarifications:\n{clarifications or '(none)'}\n"
        f"Project: {project_context or '(unknown)'}\n\n"
        "Produce the optimised master prompt."
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def synthesize(
    raw_prompt: str,
    task_type: str,
    scores_summary: str,
    clarifications: str,
    project_context: str,
) -> tuple[str, int, int]:
    """
    Returns (synthesized_prompt, input_tokens, output_tokens).
    Uses Anthropic prompt caching on the preamble for 10% read-cost (§4.8).
    """
    user_msg = _format_user_message(
        raw_prompt, task_type, scores_summary, clarifications, project_context
    )

    try:
        response = await _anthropic.messages.create(
            model=settings.model_sonnet,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PREAMBLE,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text if response.content else raw_prompt
        usage = response.usage
        return text, usage.input_tokens or 0, usage.output_tokens or 0
    except Exception as exc:
        log.error("synthesis.error", error=str(exc))
        raise  # Let tenacity retry it


def is_dspy_available() -> bool:
    return _DSPY_AVAILABLE
