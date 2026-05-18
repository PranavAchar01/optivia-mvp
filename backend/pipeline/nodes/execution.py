"""
Agent 14 — Execution Adapter (§5.3.17).

Responsibilities:
1. BUILD the mega-prompt: deterministic serialization of the task descriptor,
   agent dependency tree, per-agent system prompts, routing configuration, and
   CPM schedule into a single composite string consumed by the executor backend.
2. DELIVER that string to the pluggable executor backend (Claude Code PTY,
   subprocess, or remote API).
3. Capture the diff + exit_code + token usage emitted by the executor.
4. Emit ExecutionEvents for the Quality Monitor.

The mega-prompt format is the canonical "composite string" described in §2:
  [TASK CONTEXT]      ← synthesized master prompt (Agent 9)
  [EXECUTION CONFIG]  ← model, planning, slash-commands (Agent 10-11)
  [AGENT FLEET]       ← per-agent roles, deps, system prompts (Agent 12)
  [SCHEDULE]          ← critical path, bottlenecks (Agent 13B)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Any

import structlog

from backend.config import settings
from backend.core.models import ExecutionEvent, MasterPrompt, Outcome, RoutingDecision, WorkflowPlan
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)

_CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

# ── Mega-prompt builder ───────────────────────────────────────────────────────

def _sep(char: str = "═", width: int = 70) -> str:
    return char * width


def _build_mega_prompt(
    master_prompt: MasterPrompt,
    fleet_spec: dict[str, Any],
    routing: RoutingDecision,
    fleet_dag: dict[str, Any],
    critical_path: str,
    workflow_plan: WorkflowPlan | None,
) -> str:
    """
    Deterministic serialization of all pipeline outputs into a single composite
    string (the Mega-prompt / Master Prompt as defined in §2 Glossary).

    Structure:
      ═══ OPTIVIA ·  header ═══
      [TASK CONTEXT]
      [EXECUTION CONFIG]
      [AGENT FLEET]       — one section per sub-agent, with deps and CPM flags
      [SCHEDULE]
    """
    lines: list[str] = []
    model = routing.chosen_model
    n_agents = routing.n_agents
    complexity = fleet_dag.get("Complexity Score", "?")
    task_type = fleet_dag.get("Task Type", "unknown")
    cp = fleet_dag.get("Critical Path", critical_path or "")
    bottlenecks = set(fleet_dag.get("Bottlenecks") or [])

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(_sep())
    lines.append(
        f"OPTIVIA MEGA-PROMPT  ·  κ={complexity}  ·  {model}  ·  {n_agents} agent(s)"
    )
    lines.append(f"Task type: {task_type.upper()}")
    lines.append(_sep())
    lines.append("")

    # ── Task Context (synthesized master prompt) ───────────────────────────────
    lines.append("[TASK CONTEXT]")
    lines.append(_sep("-"))
    lines.append(master_prompt.synthesized_prompt.strip())
    lines.append("")

    # ── Execution Config ───────────────────────────────────────────────────────
    lines.append("[EXECUTION CONFIG]")
    lines.append(_sep("-"))
    lines.append(f"Model:        {model}")
    lines.append(f"Planning:     {'enabled' if routing.plan else 'off'}")
    planning_arm = routing.router_name.split(":")[-1] if ":" in routing.router_name else routing.router_name
    lines.append(f"Arm selected: {planning_arm}")
    if routing.slash_commands:
        lines.append(f"Commands:     {' '.join(routing.slash_commands)}")
    lines.append("")

    # ── Agent Fleet ──────────────────────────────────────────────────────────
    roles = fleet_spec.get("roles", [])
    dag_nodes = {n.get("Name", ""): n for n in (fleet_dag.get("Nodes") or [])}

    if not roles or roles == ["executor"]:
        # Single-agent path
        lines.append("[AGENT FLEET — single executor]")
        lines.append(_sep("-"))
        lines.append("Execute the task described above. Verify the result before finishing.")
        lines.append("")
    else:
        cp_set = set(cp.split(" -> ")) if cp else set()
        lines.append(f"[AGENT FLEET — {len(roles)} agents, dependency-ordered]")
        lines.append(_sep("-"))
        lines.append(
            "Execute agents in dependency order. Do NOT start an agent until all "
            "agents listed under 'Depends on' have completed successfully."
        )
        lines.append(
            "⚡ = on the critical path — prioritise these; delays here delay everything."
        )
        lines.append("")

        for i, role in enumerate(roles):
            if isinstance(role, str):
                lines.append(f"=== AGENT {i+1}: {role} ===")
                lines.append("Mission: Execute assigned responsibility.")
                lines.append("")
                continue

            title = role.get("title", f"Agent {i+1}")
            mission = role.get("mission", "")
            files = role.get("files", [])
            success = role.get("success", "")
            deps = role.get("dependencies", [])

            # CPM annotations from scheduler
            node_meta = dag_nodes.get(title, {})
            on_cp = node_meta.get("On Critical Path", title in cp_set)
            is_bottleneck = title in bottlenecks
            es = node_meta.get("ES", None)
            ef = node_meta.get("EF", None)

            cp_tag = " ⚡" if on_cp else ""
            bn_tag = " 🔴 BOTTLENECK" if is_bottleneck else ""
            lines.append(f"=== AGENT {i+1}: {title}{cp_tag}{bn_tag} ===")
            lines.append(f"Mission:     {mission}")
            if files:
                lines.append(f"Files:       {', '.join(files)}")
            if deps:
                lines.append(f"Depends on:  {', '.join(deps)}")
            else:
                lines.append("Depends on:  (none — start immediately)")
            lines.append(f"Success:     {success}")
            if es is not None and ef is not None:
                lines.append(f"Schedule:    ES={es}s  EF={ef}s")
            lines.append("")

    # ── Schedule summary ──────────────────────────────────────────────────────
    lines.append("[SCHEDULE]")
    lines.append(_sep("-"))
    if cp:
        lines.append(f"Critical path:  {cp}")
    cp_len = fleet_dag.get("Critical Path Length")
    if cp_len is not None:
        lines.append(f"CP wall time:   {cp_len}s estimated")
    if bottlenecks:
        lines.append(f"Bottleneck(s):  {', '.join(bottlenecks)}")
    if workflow_plan and workflow_plan.steps:
        lines.append("")
        lines.append("Execution phases:")
        for j, step in enumerate(workflow_plan.steps, 1):
            lines.append(f"  {j}. {step}")
    lines.append("")
    lines.append(_sep())

    return "\n".join(lines)


# ── Git helpers ───────────────────────────────────────────────────────────────

def _git_stash(repo_root: str) -> bool:
    try:
        r = subprocess.run(
            ["git", "stash", "push", "-u", "-m", "optivia-pre-exec"],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        return "No local changes" not in r.stdout
    except Exception:
        return False


def _git_stash_pop(repo_root: str) -> None:
    try:
        subprocess.run(["git", "stash", "pop"], capture_output=True, cwd=repo_root, timeout=10)
    except Exception:
        pass


def _git_diff_stat(repo_root: str) -> dict[str, Any]:
    try:
        r = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        lines_added = 0
        lines_removed = 0
        files_touched = 0
        for line in r.stdout.splitlines():
            if "insertion" in line or "deletion" in line:
                for part in line.strip().split(","):
                    p = part.strip()
                    if "insertion" in p:
                        lines_added = int(p.split()[0])
                    elif "deletion" in p:
                        lines_removed = int(p.split()[0])
                    elif "file" in p:
                        files_touched = int(p.split()[0])
        return {"lines_added": lines_added, "lines_removed": lines_removed, "files_touched": files_touched}
    except Exception:
        return {"lines_added": 0, "lines_removed": 0, "files_touched": 0}


# ── Subprocess execution ──────────────────────────────────────────────────────

async def _run_claude(
    mega_prompt: str,
    model: str,
    slash_commands: list[str],
    repo_root: str,
    request_id: str,
    timeout_s: int = 300,
) -> tuple[str, str, int, int]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"{settings.optivia_api_base}/proxy"
    env["OPTIVIA_REQUEST_ID"] = request_id
    env["OPTIVIA_MODEL"] = model

    cmd_prefix = " ".join(slash_commands) + "\n\n" if slash_commands else ""
    full_prompt = cmd_prefix + mega_prompt

    cmd = [_CLAUDE_CMD, "-p", full_prompt, "--model", model]

    t0 = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=repo_root,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return "", "execution timed out", 124, int((time.monotonic() - t0) * 1000)

        wall_ms = int((time.monotonic() - t0) * 1000)
        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            proc.returncode or 0,
            wall_ms,
        )
    except Exception as exc:
        return "", str(exc), 1, int((time.monotonic() - t0) * 1000)


# ── Main node ─────────────────────────────────────────────────────────────────

async def execute_via_claude_code(state: OptiviaState) -> OptiviaState:
    """Agent 14 — Execution Adapter."""
    master = state.get("master_prompt")
    routing = state.get("routing_decision")
    ctx = state.get("project_context")
    fleet_spec = state.get("fleet_spec") or {}
    fleet_dag = state.get("fleet_dag") or {}
    critical_path = state.get("critical_path", "")
    workflow_plan = state.get("workflow_plan")

    if master is None or routing is None:
        state["error"] = "missing master_prompt or routing_decision"
        return state

    # ── Build the mega-prompt (Agent 14 core responsibility) ──────────────────
    mega_prompt = _build_mega_prompt(
        master_prompt=master,
        fleet_spec=fleet_spec,
        routing=routing,
        fleet_dag=fleet_dag,
        critical_path=critical_path,
        workflow_plan=workflow_plan,
    )
    state["mega_prompt"] = mega_prompt  # type: ignore[typeddict-unknown-key]

    repo_root = (ctx.repo_root if ctx else None) or os.getcwd()
    request_id = state.get("request_id", "")
    events: list[ExecutionEvent] = list(state.get("execution_trace", []))

    # ── Simulated mode (dev / no claude binary) ───────────────────────────────
    if settings.env == "development":
        nodes = fleet_dag.get("Nodes", [])
        log.info(
            "execute.simulated",
            model=routing.chosen_model,
            n_agents=routing.n_agents,
            mega_prompt_len=len(mega_prompt),
        )
        events.append(ExecutionEvent(
            event_type="simulated_execution",
            payload={
                "model": routing.chosen_model,
                "n_agents": routing.n_agents,
                "mega_prompt_len": len(mega_prompt),
                "slash_commands": routing.slash_commands,
                "simulated_agents_dispatched": len(nodes),
                "critical_path": critical_path,
            },
            token_count=len(mega_prompt.split()) * 2,
            wall_ms=150,
        ))
        state["execution_trace"] = events
        state["outcome"] = Outcome(
            exit_code=0,
            diff_lines_added=42,
            diff_lines_removed=10,
            files_touched=len(nodes) or 1,
            user_accepted=None,
        )
        return state

    # ── Real execution ────────────────────────────────────────────────────────
    stashed = _git_stash(repo_root)

    stdout, stderr, exit_code, wall_ms = await _run_claude(
        mega_prompt=mega_prompt,
        model=routing.chosen_model,
        slash_commands=routing.slash_commands,
        repo_root=repo_root,
        request_id=request_id,
    )

    diff_stat = _git_diff_stat(repo_root)

    if exit_code != 0 and diff_stat["files_touched"] == 0 and stashed:
        _git_stash_pop(repo_root)
        log.warning("execute.rollback", exit_code=exit_code, reason="no files changed")
        events.append(ExecutionEvent(
            event_type="execution_rolled_back",
            payload={"exit_code": exit_code, "stderr": stderr[:500]},
            wall_ms=wall_ms,
        ))
    else:
        events.append(ExecutionEvent(
            event_type="claude_code_execution",
            payload={
                "exit_code": exit_code,
                "stdout_preview": stdout[:1000],
                "stderr_preview": stderr[:500],
                "mega_prompt_len": len(mega_prompt),
                **diff_stat,
            },
            token_count=state.get("action_tokens", 0),
            wall_ms=wall_ms,
        ))

    state["execution_trace"] = events
    state["action_tokens"] = state.get("action_tokens", 0) + len(stdout.split()) * 2
    state["outcome"] = Outcome(
        exit_code=exit_code,
        diff_lines_added=diff_stat["lines_added"],
        diff_lines_removed=diff_stat["lines_removed"],
        files_touched=diff_stat["files_touched"],
        user_accepted=None,
    )

    log.info(
        "execute.done",
        exit_code=exit_code,
        wall_ms=wall_ms,
        files_touched=diff_stat["files_touched"],
        mega_prompt_len=len(mega_prompt),
    )
    return state
