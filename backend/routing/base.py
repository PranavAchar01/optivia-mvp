"""Router Protocol + RoutingContext (§4.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from backend.core.models import RoutingDecision, TaskClassification, TaskScores


@dataclass(frozen=True)
class RoutingContext:
    """Everything a router needs to decide. Pure data, no I/O."""
    raw_prompt: str
    task_classification: TaskClassification
    scores: TaskScores
    master_prompt: Optional[str] = None
    workspace_id: str = ""
    user_id: str = ""


class Router(Protocol):
    """The single contract Stage 2 will swap out (§4.4)."""

    name: str

    async def route(self, ctx: RoutingContext) -> RoutingDecision: ...
