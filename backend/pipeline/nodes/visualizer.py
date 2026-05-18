"""
Agent 13 — Workflow Visualizer (§5, §6.2 node 8).

Responsibilities:
- Convert workflow_plan steps + fleet_spec into a typed React Flow graph (nodes + edges)
- Annotate each node with agent role, model tier, estimated cost
- Emit a LangGraph interrupt so the user can approve / edit before execution
- Every user edit is supervised training data (captured in the trace)
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from backend.core.models import WorkflowPlan
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)

# Visual style per task type — matches frontend NODE_COLORS
_TASK_COLOR: dict[str, str] = {
    "new_code":  "#3b82f6",
    "debug":     "#f59e0b",
    "refactor":  "#8b5cf6",
    "review":    "#10b981",
    "explain":   "#06b6d4",
    "long":      "#f97316",
    "trivial":   "#6b7280",
    "meta":      "#ec4899",
}

_MODEL_LABEL: dict[str, str] = {
    "claude-haiku-4-5":  "haiku",
    "claude-sonnet-4-6": "sonnet",
    "claude-opus-4-6":   "opus",
}


def _short_title(role: Any, fallback_index: int) -> str:
    """Extract a 2-4 word title from a FleetRole dict, str, or whatever."""
    if isinstance(role, dict):
        title = (role.get("title") or "").strip()
        if title:
            return " ".join(title.split()[:4])
    if isinstance(role, str):
        return " ".join(role.split()[:4]) or f"Agent {fallback_index + 1}"
    return f"Agent {fallback_index + 1}"


def _build_react_flow_graph(
    fleet_spec: dict[str, Any],
    task_type: str,
    routing: Any,
) -> dict[str, Any]:
    """Produce a React Flow-compatible nodes + edges payload.

    One node per agent, labeled with the role's short title. Detail
    (mission/files/success) goes into data for hover/click panels.
    """
    accent = _TASK_COLOR.get(task_type, "#3b82f6")
    model = getattr(routing, "chosen_model", "") if routing else ""
    model_label = _MODEL_LABEL.get(model, model.replace("claude-", "") if model else "?")

    roles: list[Any] = fleet_spec.get("roles", [])
    n_agents: int = fleet_spec.get("n_agents", len(roles) or 1)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for i, role in enumerate(roles[:n_agents] or [None]):
        node_id = f"agent-{i}"
        is_first = i == 0
        title = _short_title(role, i)
        detail = role if isinstance(role, dict) else {"title": title}
        nodes.append({
            "id": node_id,
            "type": "default",
            "position": {"x": 240 * i, "y": 0},
            "data": {
                "label": title,
                "mission": detail.get("mission", ""),
                "files": detail.get("files", []),
                "success": detail.get("success", ""),
                "model": model,
                "is_entry": is_first,
            },
            "style": {
                "background": accent if is_first else "#0f172a",
                "border": f"1px solid {accent if is_first else '#1e293b'}",
                "borderRadius": 8,
                "color": "#e2e8f0",
                "fontSize": 12,
                "fontFamily": "ui-monospace, monospace",
                "padding": "10px 14px",
                "maxWidth": 180,
                "fontWeight": 600,
                "boxShadow": f"0 0 12px {accent}40" if is_first else "none",
            },
        })
        if i > 0:
            edges.append({
                "id": f"e-{i-1}-{i}",
                "source": f"agent-{i-1}",
                "target": node_id,
                "animated": True,
                "style": {"stroke": accent, "strokeWidth": 1.5},
                "markerEnd": {"type": "arrowclosed", "color": accent},
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "accent": accent,
        "model_label": model_label,
    }


async def workflow_visualizer(state: OptiviaState) -> OptiviaState:
    """
    Agent 13 — Workflow Visualizer.

    Produces the React Flow graph payload and attaches it to workflow_plan.
    In Stage 1 the human-review interrupt is simulated (the API returns the plan
    and the frontend's React Flow canvas lets the user approve before /execute is called).
    Stage 2 wires this as a real LangGraph interrupt() with a resume token.
    """
    plan = state.get("workflow_plan") or WorkflowPlan()
    fleet = state.get("fleet_spec") or {}
    task_cls = state.get("task_classification")
    routing = state.get("routing_decision")

    task_type = task_cls.task_type.value if task_cls else "new_code"

    graph_payload = _build_react_flow_graph(
        fleet_spec=fleet,
        task_type=task_type,
        routing=routing,
    )

    # Keep plan.steps as the human-readable summary (used by master prompt context),
    # but visualizer_json now drives the frontend rendering with short titles.
    state["workflow_plan"] = WorkflowPlan(
        steps=plan.steps or [n.get("data", {}).get("label", "") for n in graph_payload["nodes"]],
        visualizer_json=graph_payload,
    )

    # Stage 1: mark as human-reviewed (auto-approved)
    # Stage 2: interrupt() here and wait for resume signal from the frontend
    state["_plan_approved"] = True  # type: ignore[typeddict-unknown-key]
    state["_plan_edited_by_user"] = False  # type: ignore[typeddict-unknown-key]

    log.info(
        "workflow_visualizer.done",
        n_nodes=len(graph_payload["nodes"]),
        n_edges=len(graph_payload["edges"]),
        task_type=task_type,
    )
    return state
