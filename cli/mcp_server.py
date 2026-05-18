"""
Optivia MCP server (§6.5 Surface 2).

Claude Code users install this via:
    claude mcp add optivia python -m cli.mcp_server

The server exposes one tool, `optimize_prompt`, that takes a raw developer
prompt and returns Optivia's optimised master prompt + workflow plan +
routing decision. The user can then feed the master prompt back into
Claude Code as the next user turn.

This is the *lowest-friction* surface: it doesn't require ANTHROPIC_BASE_URL
or a subprocess wrapper. The trade-off is that Optivia doesn't see the
downstream Claude Code execution, so the trace contract is partial.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

_API_BASE = os.environ.get("OPTIVIA_API_BASE", "http://localhost:8000")
server = Server("optivia")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="optimize_prompt",
            description=(
                "Convert a vague developer prompt into an Optivia-optimised master "
                "prompt with classification, complexity score, sub-agent plan, and "
                "routing decision. Returns JSON with master_prompt, model, n_agents, "
                "slash_commands, workflow_plan, complexity, and task_type."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The raw, possibly-vague coding task",
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Workspace / project id (optional)",
                    },
                    "user": {
                        "type": "string",
                        "description": "User id (optional)",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="get_trace",
            description="Fetch a previously-generated trace by trace_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string"},
                },
                "required": ["trace_id"],
            },
        ),
        Tool(
            name="submit_feedback",
            description="Submit thumbs feedback for a previously-generated trace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string"},
                    "thumbs": {"type": "integer", "description": "-1, 0, or 1"},
                    "followup": {"type": "string", "description": "Optional follow-up prompt"},
                },
                "required": ["trace_id", "thumbs"],
            },
        ),
    ]


async def _call_optimize(prompt: str, workspace: str, user: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{_API_BASE}/optimize",
            json={
                "prompt": prompt,
                "workspace_id": workspace or "",
                "user_id": user or "",
            },
        )
        r.raise_for_status()
        return r.json()


async def _call_trace(trace_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{_API_BASE}/trace/{trace_id}")
        r.raise_for_status()
        return r.json()


async def _call_feedback(trace_id: str, thumbs: int, followup: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{_API_BASE}/feedback",
            json={
                "trace_id": trace_id,
                "thumbs": thumbs,
                "followup_prompt": followup or None,
            },
        )
        r.raise_for_status()
        return r.json()


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "optimize_prompt":
            result = await _call_optimize(
                arguments.get("prompt", ""),
                arguments.get("workspace", ""),
                arguments.get("user", ""),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        if name == "get_trace":
            result = await _call_trace(arguments.get("trace_id", ""))
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        if name == "submit_feedback":
            result = await _call_feedback(
                arguments.get("trace_id", ""),
                int(arguments.get("thumbs", 0)),
                arguments.get("followup", ""),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        return [TextContent(type="text", text=f"unknown tool: {name}")]
    except httpx.HTTPError as exc:
        return [TextContent(
            type="text",
            text=(
                f"Optivia backend error ({_API_BASE}): {exc}. "
                "Make sure `optivia serve` is running."
            ),
        )]
    except Exception as exc:
        return [TextContent(type="text", text=f"error: {exc}")]


async def main_async() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
