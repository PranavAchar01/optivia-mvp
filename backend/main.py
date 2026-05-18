"""FastAPI application — Optivia Stage 1 Wrapper MVP."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import structlog
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.core.models import Outcome, ProjectContext, UserFeedback
from backend.db.client import db_client
from backend.pipeline.graph import pipeline
from backend.pipeline.state import OptiviaState
from backend.proxy import register_master_prompt, router as proxy_router

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db_client.connect()
    except Exception as exc:
        log.warning("db.connect_failed", error=str(exc))
    yield
    await db_client.disconnect()


app = FastAPI(
    title="Optivia",
    description="Pre-execution optimization layer for agentic coding CLIs",
    version="0.1.0",
    lifespan=lifespan,
)

_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "tauri://localhost",
    "https://tauri.localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"^(https?|tauri)://(localhost|127\.0\.0\.1|tauri\.localhost)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic API proxy — Claude Code points ANTHROPIC_BASE_URL here (§6.5)
app.include_router(proxy_router)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    prompt: str
    user_id: str = ""
    workspace_id: str = ""
    project_context: Optional[ProjectContext] = None
    session_id: Optional[str] = None
    clarification_answers: Optional[list[str]] = None  # provided if continuing a clarification


class ContinueRequest(BaseModel):
    request_id: str
    answers: list[str]
    user_id: str = ""
    workspace_id: str = ""


class ClarificationResponse(BaseModel):
    request_id: str
    questions: list[dict[str, Any]]
    requires_clarification: bool = True


class OptimizeResponse(BaseModel):
    Task_Type: str
    Complexity_Score: int
    Environment_Target: str
    Nodes: list[dict[str, Any]]
    Edges: list[dict[str, Any]]
    Critical_Path: str
    
    # Keeping old fields for compatibility, making them optional
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    master_prompt: Optional[str] = None
    model: Optional[str] = None
    n_agents: Optional[int] = None
    slash_commands: Optional[list[str]] = None
    workflow_plan: Optional[list[str]] = None
    complexity: Optional[int] = None
    specificity: Optional[float] = None
    task_type: Optional[str] = None
    requires_clarification: bool = False
    clarification_questions: list[dict[str, Any]] = []

    class Config:
        alias_generator = lambda string: string.replace('_', ' ')
        populate_by_name = True


class FeedbackRequest(BaseModel):
    trace_id: str
    thumbs: int  # -1 / 0 / 1
    followup_prompt: Optional[str] = None


def _extract_plan_labels(plan: Any) -> list[str]:
    """Return short agent titles for the frontend visualizer.

    Prefers visualizer_json node labels (short titles like "ORM Migrator"),
    which live at node["data"]["label"]. Falls back to plan.steps.
    """
    if not plan:
        return []
    viz = getattr(plan, "visualizer_json", None) or {}
    nodes = viz.get("nodes", []) if isinstance(viz, dict) else []
    labels: list[str] = []
    for n in nodes:
        data = n.get("data") if isinstance(n, dict) else None
        label = (data or {}).get("label") or n.get("label", "")
        label = str(label).strip()
        if label:
            labels.append(label)
    if labels:
        return labels
    return list(getattr(plan, "steps", []) or [])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest) -> OptimizeResponse:
    """
    Main entry point — runs the 17-agent pipeline and returns the optimised
    master prompt, routing decision, and workflow plan.

    If clarification is needed and no answers provided, returns 202 with questions.
    """
    initial_state: OptiviaState = {
        "request_id": str(uuid.uuid4()),
        "user_id": req.user_id or "00000000-0000-0000-0000-000000000000",
        "workspace_id": req.workspace_id or "00000000-0000-0000-0000-000000000000",
        "raw_prompt": req.prompt,
        "attached_files": [],
        "project_context": req.project_context or ProjectContext(),
        "clarifications": [],
        "clarification_round": 0,
        "consecutive_high_quality": 0,
        "execution_trace": [],
        "adaptation_actions": [],
        "obs_tokens": 0,
        "memory_tokens": 0,
        "plan_tokens": 0,
        "action_tokens": 0,
        "turn_index": 0,
    }

    try:
        result: OptiviaState = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        log.error("optimize.pipeline_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    # Check if clarification was requested (no answers yet)
    clarifications = result.get("clarifications", [])
    pending_clarification = (
        clarifications
        and clarifications[-1].questions
        and not clarifications[-1].answers
    )

    request_id_local = result.get("request_id", "")
    if pending_clarification and request_id_local:
        await _save_pending(request_id_local, {
            "raw_prompt": req.prompt,
            "user_id": req.user_id,
            "workspace_id": req.workspace_id,
            "project_context": (req.project_context.model_dump() if req.project_context else {}),
            "questions": [q.model_dump() for q in clarifications[-1].questions],
            "turn_index": result.get("turn_index", 0),
        })

    master = result.get("master_prompt")
    routing = result.get("routing_decision")
    scores = result.get("scores_updated") or result.get("scores")
    task_cls = result.get("task_classification")
    plan = result.get("workflow_plan")
    fleet_dag = result.get("fleet_dag") or {}

    # Agent 14 builds the mega-prompt; fall back to synthesized_prompt if pipeline
    # was short-circuited (cache hit / fast-intent) before reaching Agent 14.
    mega_prompt: str = (
        result.get("mega_prompt")  # type: ignore[typeddict-item]
        or (master.synthesized_prompt if master else req.prompt)
    )

    # Register with the proxy so the first Claude Code call gets the mega-prompt
    # injected as a cached system message (§6.5).
    request_id = result.get("request_id", "")
    if request_id and mega_prompt:
        register_master_prompt(request_id, mega_prompt)

    return OptimizeResponse(
        Task_Type=fleet_dag.get("Task Type", task_cls.task_type.value if task_cls else "unknown"),
        Complexity_Score=fleet_dag.get("Complexity Score", scores.complexity if scores else 5),
        Environment_Target=fleet_dag.get("Environment Target", "Claude Code"),
        Nodes=fleet_dag.get("Nodes", []),
        Edges=fleet_dag.get("Edges", []),
        Critical_Path=fleet_dag.get("Critical Path", ""),
        request_id=request_id,
        trace_id=result.get("trace_id", ""),
        master_prompt=mega_prompt,
        model=routing.chosen_model if routing else settings.model_sonnet,
        n_agents=routing.n_agents if routing else 1,
        slash_commands=routing.slash_commands if routing else [],
        workflow_plan=_extract_plan_labels(plan),
        complexity=scores.complexity if scores else 5,
        specificity=scores.specificity if scores else 0.5,
        task_type=task_cls.task_type.value if task_cls else "unknown",
        requires_clarification=bool(pending_clarification),
        clarification_questions=[
            {"dimension": q.dimension, "question": q.question}
            for q in (clarifications[-1].questions if pending_clarification else [])
        ],
    )


_CLARIFICATION_TTL = 1800  # 30 minutes


async def _redis():
    import redis.asyncio as aioredis
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def _save_pending(request_id: str, data: dict[str, Any]) -> None:
    try:
        r = await _redis()
        async with r:
            await r.setex(
                f"clarif:{request_id}", _CLARIFICATION_TTL, json.dumps(data)
            )
    except Exception as exc:
        log.warning("main.save_pending_error", error=str(exc))


async def _pop_pending(request_id: str) -> dict[str, Any] | None:
    try:
        r = await _redis()
        async with r:
            raw = await r.getdel(f"clarif:{request_id}")
            return json.loads(raw) if raw else None
    except Exception as exc:
        log.warning("main.pop_pending_error", error=str(exc))
        return None


@app.post("/optimize/continue", response_model=OptimizeResponse)
async def optimize_continue(req: ContinueRequest) -> OptimizeResponse:
    """
    Resumes a pipeline run that paused for clarification. The caller passes
    back the request_id and the answers; we re-run the pipeline with the
    answers stitched into the clarifications state so the re-scorer and
    synthesiser see them.
    """
    saved = await _pop_pending(req.request_id)
    if not saved:
        raise HTTPException(status_code=404, detail="no pending clarification for that request")

    # Stitch answers into the saved clarification
    from backend.core.models import Clarification, ClarificationQuestion

    questions = [
        ClarificationQuestion(**q) for q in saved["questions"]
    ]
    answered = Clarification(
        questions=questions,
        answers=list(req.answers),
        sufficiency_passed=True,
    )

    initial_state: OptiviaState = {
        "request_id": req.request_id,
        "user_id": req.user_id or saved.get("user_id", "anonymous"),
        "workspace_id": req.workspace_id or saved.get("workspace_id", ""),
        "raw_prompt": saved["raw_prompt"],
        "attached_files": [],
        "project_context": ProjectContext(**(saved.get("project_context") or {})),
        "clarifications": [answered],
        "clarification_round": 1,  # mark as already-clarified so the gate skips re-clarification
        "consecutive_high_quality": 0,
        "execution_trace": [],
        "adaptation_actions": [],
        "obs_tokens": 0,
        "memory_tokens": 0,
        "plan_tokens": 0,
        "action_tokens": 0,
        "turn_index": saved.get("turn_index", 0) + 1,
    }

    try:
        result: OptiviaState = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        log.error("optimize_continue.pipeline_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    master = result.get("master_prompt")
    routing = result.get("routing_decision")
    scores = result.get("scores_updated") or result.get("scores")
    task_cls = result.get("task_classification")
    plan = result.get("workflow_plan")
    fleet_dag = result.get("fleet_dag") or {}

    mega_prompt: str = (
        result.get("mega_prompt")  # type: ignore[typeddict-item]
        or (master.synthesized_prompt if master else saved["raw_prompt"])
    )

    request_id = result.get("request_id", "")
    if request_id and mega_prompt:
        register_master_prompt(request_id, mega_prompt)

    return OptimizeResponse(
        Task_Type=fleet_dag.get("Task Type", task_cls.task_type.value if task_cls else "unknown"),
        Complexity_Score=fleet_dag.get("Complexity Score", scores.complexity if scores else 5),
        Environment_Target=fleet_dag.get("Environment Target", "Claude Code"),
        Nodes=fleet_dag.get("Nodes", []),
        Edges=fleet_dag.get("Edges", []),
        Critical_Path=fleet_dag.get("Critical Path", ""),
        request_id=request_id,
        trace_id=result.get("trace_id", ""),
        master_prompt=mega_prompt,
        model=routing.chosen_model if routing else settings.model_sonnet,
        n_agents=routing.n_agents if routing else 1,
        slash_commands=routing.slash_commands if routing else [],
        workflow_plan=_extract_plan_labels(plan),
        complexity=scores.complexity if scores else 5,
        specificity=scores.specificity if scores else 0.5,
        task_type=task_cls.task_type.value if task_cls else "unknown",
        requires_clarification=False,
        clarification_questions=[],
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict[str, str]:
    """Records user feedback (thumbs up/down) for a completed trace."""
    fb = UserFeedback(
        thumbs=req.thumbs,
        followup_prompt=req.followup_prompt,
    )
    try:
        trace = await db_client.get_trace(req.trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="trace not found")
        await db_client.update_trace_feedback(req.trace_id, fb.model_dump())
        log.info("feedback.recorded", trace_id=req.trace_id, thumbs=req.thumbs)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("feedback.error", error=str(exc))

    return {"status": "ok"}


@app.get("/traces/{workspace_id}")
async def list_traces(workspace_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Returns recent traces for a workspace (observability dashboard feed)."""
    traces = await db_client.get_session_traces(workspace_id, limit=limit)
    return traces


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """Returns a single trace by ID."""
    trace = await db_client.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="not found")
    return trace


# ---------------------------------------------------------------------------
# Internal endpoint — called by the CLI proxy (§6.5)
# ---------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    request_id: str
    master_prompt: str
    model: str
    n_agents: int
    slash_commands: list[str]
    workflow_plan: list[str]


@app.post("/internal/execute")
async def internal_execute(req: ExecuteRequest) -> dict[str, Any]:
    """
    Called by the CLI execution adapter. In Stage 1 this is a pass-through
    that logs the event. In production it manages the Claude Code subprocess.
    """
    log.info(
        "internal.execute",
        request_id=req.request_id,
        model=req.model,
        n_agents=req.n_agents,
    )
    return {
        "status": "dispatched",
        "request_id": req.request_id,
        "tokens_used": 0,
    }


# ---------------------------------------------------------------------------
# /internal/observe — Claude Code PostToolUse hook posts events here
# ---------------------------------------------------------------------------

class ObserveEvent(BaseModel):
    request_id: str = ""
    trace_id: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = {}
    tool_response: dict[str, Any] = {}


@app.post("/internal/observe")
async def internal_observe(req: ObserveEvent) -> dict[str, Any]:
    """Receives Claude Code hook events for downstream observability."""
    log.info(
        "internal.observe",
        request_id=req.request_id,
        trace_id=req.trace_id,
        tool_name=req.tool_name,
    )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /metrics — nightly evaluation aggregates (§4.10)
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def metrics(workspace_id: str | None = None) -> dict[str, Any]:
    """
    Aggregate metrics for the nightly eval job: total traces, avg complexity,
    router-mix distribution. The Modal scheduled function calls this and
    pushes the result to Slack.
    """
    return await db_client.aggregate_metrics(workspace_id)
