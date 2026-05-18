"""
Optivia CLI — Python typer wrapper for the Optivia Engine (v15 Architecture).

Usage:
    optivia run "build a login system with Supabase"
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional

import typer
from rich.console import Console

from backend.core.models import ProjectContext
from backend.pipeline.graph import pipeline
from backend.pipeline.state import OptiviaState
from backend.db.client import db_client

app = typer.Typer(name="optivia", help="Optivia Engine: Pre-execution optimization layer and dynamic agent fleet generator")
console = Console()


async def _execute_pipeline(state: OptiviaState) -> OptiviaState:
    # Attempt DB connection for cached lookups (fail gracefully if not available locally)
    try:
        await db_client.connect()
    except Exception:
        pass
        
    try:
        result = await pipeline.ainvoke(state)
        return result
    finally:
        try:
            await db_client.disconnect()
        except Exception:
            pass


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The raw user prompt to process"),
    workspace: str = typer.Option("", "--workspace", "-w", help="Workspace / project name"),
    user: str = typer.Option("", "--user", "-u", help="User ID"),
) -> None:
    """Generate an Optivia agent fleet JSON DAG based on the prompt."""
    
    # Suppress verbose pipeline logs from structlog to keep output clean for the DAG
    import logging
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))
    
    console.print(f"[bold cyan]Optivia Engine[/] analyzing prompt and generating agent fleet...")

    initial_state: OptiviaState = {
        "request_id": str(uuid.uuid4()),
        "user_id": user or "00000000-0000-0000-0000-000000000000",
        "workspace_id": workspace or "00000000-0000-0000-0000-000000000000",
        "raw_prompt": prompt,
        "attached_files": [],
        "project_context": ProjectContext(),
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
        result = asyncio.run(_execute_pipeline(initial_state))
    except Exception as exc:
        console.print(f"[red]Pipeline execution failed:[/] {exc}")
        raise typer.Exit(1)

    if result.get("error"):
        console.print(f"[red]Error:[/] {result['error']}")
        raise typer.Exit(1)

    fleet_dag = result.get("fleet_dag", {})
    if not fleet_dag:
        console.print("[yellow]Warning:[/] No fleet DAG generated. Check the prompt complexity and constraints.")
        raise typer.Exit(1)

    # Output the optimized, fully synthesized agent fleet DAG
    console.print_json(json.dumps(fleet_dag))


if __name__ == "__main__":
    app()
