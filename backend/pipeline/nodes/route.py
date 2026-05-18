"""Nodes 10–12 — Routing Engine + Conflict Resolver + Fleet Generator (§4.7–4.8, §6.2 nodes 7,9)."""

from __future__ import annotations

import structlog
import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from backend.config import settings
from backend.core.models import WorkflowPlan
from backend.pipeline.state import OptiviaState
from backend.routing import RoutingContext, get_router_runner
from backend.routing.bandit import BanditEnsemble, build_context_vector
from backend.routing.safety_router import arm_catalog, get_allowed_arms

log = structlog.get_logger(__name__)

client = instructor.from_anthropic(AsyncAnthropic(api_key=settings.anthropic_api_key))


# ---------------------------------------------------------------------------
# Node 10+11: route + conflict_resolver — shadow routing across 3 routers (§4.4)
# ---------------------------------------------------------------------------

async def route_and_resolve(state: OptiviaState) -> OptiviaState:
    """
    Runs Heuristic + RouteLLM-mf + LLMJudge in parallel.
    The active router's decision becomes routing_decision; the others are
    stashed on state for the persister to write to routing_decisions table.
    """
    scores = state.get("scores_updated") or state.get("scores")
    task_cls = state.get("task_classification")
    master = state.get("master_prompt")

    if scores is None or task_cls is None:
        state["error"] = "missing scores or classification for routing"
        return state

    ctx = RoutingContext(
        raw_prompt=state.get("raw_prompt", ""),
        task_classification=task_cls,
        scores=scores,
        master_prompt=master.synthesized_prompt if master else None,
        workspace_id=state.get("workspace_id", ""),
        user_id=state.get("user_id", ""),
    )

    runner = get_router_runner()
    heuristic_active, shadow_rows = await runner.run(ctx)

    # ── Agent 10A: deterministic safety filter ───────────────────────────────
    allowed_arms = get_allowed_arms(
        risk=scores.risk,
        complexity=scores.complexity,
        task_type=task_cls.task_type,
        budget_rho=state.get("token_budget_rho", 0.0),
        remaining_budget_usd=None,
        policy_require_verifier=False,
    )
    state["allowed_arms"] = [a.arm_id for a in allowed_arms]  # type: ignore[typeddict-unknown-key]

    # ── Agent 10B: D-LinUCB selection over allowed arms ──────────────────────
    workspace_id = state.get("workspace_id", "")
    ensemble = await BanditEnsemble.load(workspace_id)

    x = build_context_vector(
        scores=scores,
        task_type=task_cls.task_type,
        budget_rho=state.get("token_budget_rho", 0.0),
        avg_quality=0.8,            # seeded; updated when persister sees prior turn
        retry_rate=min(1.0, state.get("clarification_round", 0) / 3.0),
        executor="claude_code",
        context_size_norm=min(1.0, len(state.get("raw_prompt", "")) / 4000.0),
        cpl_norm=state.get("cpl_norm", 0.0),  # type: ignore[typeddict-item]
    )

    # Cold-start fallback: the κ-driven heuristic arm
    catalog = arm_catalog()
    fallback_arm = next(
        (a for a in catalog if a.model == heuristic_active.chosen_model),
        allowed_arms[0],
    )
    if fallback_arm not in allowed_arms:
        fallback_arm = allowed_arms[0]

    chosen_arm, ucb_score, arm_scores = ensemble.select(
        x, allowed_arms, cold_start_fallback=fallback_arm,
    )

    # Build the routing decision from the chosen arm (model + n_agents + planning)
    active = heuristic_active.model_copy()
    active.chosen_model = chosen_arm.model
    active.n_agents = max(chosen_arm.n_agents, active.n_agents)
    active.plan = "planning_enabled" if chosen_arm.planning_on else ""
    active.router_name = f"d_linucb:{chosen_arm.arm_id}"
    active.router_score = ucb_score
    active.alternatives = [
        {"arm_id": aid, "ucb": round(float(s), 4)} for aid, s in arm_scores.items()
    ]

    state["routing_decision"] = active
    state["_shadow_routing_rows"] = shadow_rows  # type: ignore[typeddict-unknown-key]
    state["bandit_arm_selected"] = chosen_arm.arm_id  # type: ignore[typeddict-unknown-key]
    state["_bandit_context_vector"] = x.tolist()  # type: ignore[typeddict-unknown-key]

    state["model_tier_decisions"] = {
        "selected_arm": chosen_arm.arm_id,
        "selected_tier": chosen_arm.model_tier,
        "selected_model": chosen_arm.model,
        "ucb_score": round(float(ucb_score), 4),
        "alternatives": active.alternatives,
        "allowed_arms": [a.arm_id for a in allowed_arms],
    }

    log.info(
        "route.done",
        arm=chosen_arm.arm_id,
        active_model=active.chosen_model,
        n_agents=active.n_agents,
        commands=active.slash_commands,
        shadow_routers=[r["router_name"] for r in shadow_rows],
        allowed_arms=[a.arm_id for a in allowed_arms],
    )
    return state


# ---------------------------------------------------------------------------
# Node 12: sub_agent_fleet_generator
# ---------------------------------------------------------------------------

class FleetRole(BaseModel):
    title: str       # short label, 2-4 words (e.g. "ORM Migrator", "Token Service")
    mission: str    # one-sentence mission
    files: list[str]  # files this agent touches
    success: str    # one-line success criterion
    dependencies: list[str] = []  # titles of upstream agents (must complete first)


class FleetSpec(BaseModel):
    roles: list[FleetRole]
    n_agents: int
    orchestration_model: str
    worker_model: str
    steps: list[str]


_FLEET_SYSTEM = """\
You are Optivia's Sub-Agent Fleet Generator. Design a fragmented sub-agent fleet
for Claude Code that hits the n_agents target.

Rules:
- Each role gets exactly ONE responsibility — one file, one layer, one concern.
- `title` MUST be 2-4 words, e.g. "ORM Migrator", "Route Handler", "Test Author".
  Do NOT put instructions in the title. Save those for `mission`/`success`.
- `mission` is one sentence describing what this agent does.
- `files` lists the specific files this agent owns (relative paths).
- `success` is the single check that proves this agent finished.
- `dependencies` lists the titles of upstream agents whose work MUST complete
  before this agent can start. Use [] for roots. The whole collection MUST form
  an acyclic DAG (no cycles, no forward references).
- Prefer many small agents over few large ones. Split by file boundary,
  by layer (schema/service/route/test), or by phase (design/implement/verify).
- Prefer fan-out: many independent leaves at the same depth instead of a long
  linear chain. Tests/audits typically depend on implementer agents.
- For κ≥7 fleets, always include: 1 design/audit agent, 1+ test agents,
  and parallel implementer agents — one per file or layer.
"""


def _validate_dag(roles: list[FleetRole]) -> list[FleetRole]:
    """Drop unknown deps; break any cycles via topological repair."""
    title_set = {r.title for r in roles}
    for r in roles:
        r.dependencies = [d for d in r.dependencies if d in title_set and d != r.title]

    # Iterative Kahn's algorithm — anything left in `unvisited` after the loop
    # is part of a cycle and has its deps cleared.
    in_deg = {r.title: len(r.dependencies) for r in roles}
    by_title = {r.title: r for r in roles}
    ready = [t for t, n in in_deg.items() if n == 0]
    visited: set[str] = set()
    while ready:
        t = ready.pop(0)
        visited.add(t)
        for cand in roles:
            if t in cand.dependencies:
                in_deg[cand.title] -= 1
                if in_deg[cand.title] == 0:
                    ready.append(cand.title)
    for t in set(by_title) - visited:
        by_title[t].dependencies = []  # cycle break
    return roles


def _build_dag_edges(roles: list[FleetRole]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for r in roles:
        for dep in r.dependencies:
            edges.append({"From": dep, "To": r.title})
    return edges


async def fleet_generator(state: OptiviaState) -> OptiviaState:
    """Node: sub_agent_fleet_generator (agent 12 in §5)."""
    routing = state.get("routing_decision")
    master = state.get("master_prompt")
    task_cls = state.get("task_classification")
    scores = state.get("scores_updated") or state.get("scores")

    if routing is None or master is None:
        return state

    n_agents = routing.n_agents
    if n_agents <= 1:
        model = routing.chosen_model if routing else settings.model_haiku
        mission = master.synthesized_prompt[:300] if master else "Execute the task."
        state["fleet_spec"] = {
            "roles": [{"title": "Executor", "mission": mission, "files": [], "success": "Task completed successfully.", "dependencies": []}],
            "n_agents": 1,
            "orchestration_model": model,
            "worker_model": model,
            "steps": ["Execute task with master prompt"],
        }
        state["fleet_dag"] = {
            "Task Type": task_cls.task_type.value if task_cls else "unknown",
            "Complexity Score": scores.complexity if scores else 1,
            "Environment Target": "Claude Code",
            "Nodes": [
                {
                    "Name": "Executor",
                    "Model Tier": model,
                    "System Prompt": mission,
                    "Estimated Tokens": 1000,
                    "Estimated Duration": 30.0,
                    "On Critical Path": True,
                    "Bottleneck": False,
                    "ES": 0.0, "EF": 30.0, "LS": 0.0, "LF": 30.0, "Slack": 0.0,
                }
            ],
            "Edges": [],
            "Critical Path": "Executor",
            "Critical Path Length": 30.0,
            "Bottlenecks": [],
        }
        state["workflow_plan"] = WorkflowPlan(
            steps=["Execute task with master prompt"],
            visualizer_json={
                "nodes": [{"id": "exec", "label": "Executor", "mission": mission, "files": [], "success": "Task completed.", "dependencies": []}],
                "edges": [],
            },
        )
        return state

    try:
        result: FleetSpec = await client.chat.completions.create(
            model=settings.model_sonnet,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Task type: {task_cls.task_type.value if task_cls else 'unknown'}\n"
                        f"Complexity κ={scores.complexity if scores else '?'}\n"
                        f"Scope δ_s={scores.scope if scores else '?'} | "
                        f"Dependency δ_d={scores.dependency if scores else '?'}\n"
                        f"n_agents target: {n_agents} (you MUST produce exactly this many roles)\n\n"
                        f"Master prompt:\n{master.synthesized_prompt[:1200]}\n\n"
                        f"Decompose into {n_agents} fragmented agents. Each agent owns one "
                        f"concrete piece — single file, single layer, or single phase."
                    ),
                }
            ],
            system=_FLEET_SYSTEM,
            response_model=FleetSpec,
        )

        result.roles = _validate_dag(result.roles)
        state["fleet_spec"] = result.model_dump()

        title_to_id = {r.title: f"agent_{i}" for i, r in enumerate(result.roles)}

        nodes = []
        for r in result.roles:
            nodes.append({
                "Name": r.title,
                "Model Tier": routing.chosen_model,
                "System Prompt": r.mission,
                "Estimated Tokens": 1500,
                "Estimated Duration": 30.0,  # seconds — CPM input, overridable by history
            })

        dag_edges = _build_dag_edges(result.roles)

        state["fleet_dag"] = {
            "Task Type": task_cls.task_type.value if task_cls else "unknown",
            "Complexity Score": scores.complexity if scores else 5,
            "Environment Target": "Claude Code",
            "Nodes": nodes,
            "Edges": dag_edges,
        }

        state["workflow_plan"] = WorkflowPlan(
            steps=result.steps,
            visualizer_json={
                "nodes": [
                    {
                        "id": title_to_id[r.title],
                        "label": r.title,
                        "mission": r.mission,
                        "files": r.files,
                        "success": r.success,
                        "dependencies": r.dependencies,
                    }
                    for r in result.roles
                ],
                "edges": [
                    {
                        "from": title_to_id[dep],
                        "to": title_to_id[r.title],
                    }
                    for r in result.roles for dep in r.dependencies
                    if dep in title_to_id
                ],
            },
        )
    except Exception as exc:
        log.error("fleet_generator.error", error=str(exc))
        state["fleet_spec"] = {"roles": ["executor"], "n_agents": 1}
        state["workflow_plan"] = WorkflowPlan(steps=["Execute task"])
        state["fleet_dag"] = {}

    return state
