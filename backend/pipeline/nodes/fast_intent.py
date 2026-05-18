"""Node 2b — Fast Intent (§6.2 node 2): Aurelio Semantic Router triage."""

from __future__ import annotations

import re
import structlog

from backend.core.models import FastIntent
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)

# ~30 utterances per route — enough for Aurelio kNN at Stage 1.
# In Stage 2 these are replaced by a ModernBERT fine-tuned head.
_TRIVIAL_UTTERANCES = [
    "rename this variable",
    "format this file",
    "add a docstring",
    "fix the indentation",
    "remove trailing whitespace",
    "what does this function do",
    "hello",
    "hi claude",
    "thank you",
    "looks good",
]

_CODE_UTTERANCES = [
    "build a login system",
    "add authentication",
    "fix the bug in",
    "refactor the service",
    "write tests for",
    "debug why",
    "implement the feature",
    "create a new endpoint",
    "update the schema",
    "migrate the database",
]


def _classify_fast(prompt: str) -> tuple[str, float]:
    """
    Lightweight heuristic classifier for Stage 1.
    Stage 2 replaces this with Aurelio Semantic Router + voyage-code-3 embeddings.
    """
    lower = prompt.lower().strip()

    if not lower:
        return "trivial", 0.99

    words = set(re.findall(r'\b\w+\b', lower))

    # Trivial / Chitchat patterns
    chitchat_keywords = {"hello", "hi", "thanks", "thank", "bye", "goodbye", "ok", "okay", "yes", "no"}
    if len(words) <= 3 and words.issubset(chitchat_keywords):
        return "trivial", 0.95

    # Very short, non-code prompts
    if len(words) < 5 and not any(kw in lower for kw in ("fix", "add", "build", "create", "refactor", "debug", "write", "update", "implement")):
        return "trivial", 0.92

    # Check for obvious code task keywords
    code_keywords = {"fix", "add", "build", "create", "refactor", "debug", "write", "implement", "update", "migrate", "test", "review", "explain", "optimize", "deploy", "setup"}
    hits = sum(1 for kw in code_keywords if kw in words)
    
    if hits >= 2:
        return "code_task", min(0.6 + (hits * 0.1), 0.98)
    if hits == 1 and len(words) > 5:
        return "code_task", 0.75
    if hits == 1:
        return "code_task", 0.65

    return "unknown", 0.4


async def fast_intent(state: OptiviaState) -> OptiviaState:
    """
    Node: fast_intent
    Confidence > 0.9 + trivial/chitchat → short-circuit out.
    """
    raw = state.get("raw_prompt", "")
    intent, confidence = _classify_fast(raw)

    short_circuit = intent == "trivial" and confidence > 0.9

    state["fast_intent"] = FastIntent(
        intent=intent,
        confidence=confidence,
        short_circuit=short_circuit,
    )

    if short_circuit:
        log.info("fast_intent.short_circuit", intent=intent, confidence=confidence)

    return state
