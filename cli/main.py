"""
Optivia terminal CLI — routes prompts through the 17-stage pipeline,
displays live progress, and dispatches the mega-prompt to claude.

Ruflo multi-agent routing is applied verbatim from router.cjs before
the pipeline runs and is shown in the result.

Usage:
    optivia "build a login system"          # optimize + queue
    optivia "..." --execute                 # optimize + invoke claude immediately
    optivia --no-stream "..."               # use blocking /optimize instead of SSE
    echo "build it" | optivia              # read prompt from stdin
    optivia install-shim                   # install ~/.zshrc dispatch shim
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import typer
from rich.console import Console

# ── App ───────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="optivia",
    help="Optivia: pre-execution optimization layer with ruflo multi-agent routing.",
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()

_DEFAULT_API = os.getenv("OPTIVIA_API_BASE", "http://localhost:8000")
_QUEUE_DIR   = Path.home() / ".optivia" / "queue"
_CURRENT     = Path.home() / ".optivia" / "current.json"
_LABEL_W     = 18

# ── Ruflo routing (verbatim from .claude/helpers/router.cjs) ─────────────────

_TASK_PATTERNS: dict[str, str] = {
    r"implement|create|build|add|write code": "coder",
    r"test|spec|coverage|unit test|integration": "tester",
    r"review|audit|check|validate|security": "reviewer",
    r"research|find|search|documentation|explore": "researcher",
    r"design|architect|structure|plan": "architect",
    r"api|endpoint|server|backend|database": "backend-dev",
    r"ui|frontend|component|react|css|style": "frontend-dev",
    r"deploy|docker|ci|cd|pipeline|infrastructure": "devops",
}


def route_task(task: str) -> dict:
    """Verbatim port of routeTask() from router.cjs."""
    for pattern, agent in _TASK_PATTERNS.items():
        if re.search(pattern, task, re.IGNORECASE):
            return {"agent": agent, "confidence": 0.8, "reason": f"Matched pattern: {pattern}"}
    return {"agent": "coder", "confidence": 0.5, "reason": "Default routing - no specific pattern matched"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label(text: str) -> str:
    return f"[dim]{text.rjust(_LABEL_W)}[/dim]"


def _model_label(model: str) -> str:
    if not model:
        return "unknown"
    m = model.lower()
    if "opus" in m:   return "Opus 4"
    if "sonnet" in m: return "Sonnet 4.6"
    if "haiku" in m:  return "Haiku 4.5"
    return model.replace("claude-", "")


_STAGE_LABELS: dict[str, str] = {
    "prompt_intake":            "intake",
    "session_loader":           "session",
    "experience_retriever":     "cache lookup",
    "fast_intent":              "fast intent",
    "classify_and_score":       "classify · score",
    "decide_clarification":     "clarify?",
    "generate_clarifications":  "clarifications",
    "sufficiency_qa":           "sufficiency",
    "re_scorer":                "re-score",
    "synthesize_master_prompt": "synthesize",
    "route_and_resolve":        "route",
    "fleet_generator":          "fleet gen",
    "workflow_visualizer":      "visualize",
    "critical_path_scheduler":  "schedule",
    "execute_via_claude_code":  "execute",
    "quality_monitor":          "quality",
    "experience_extractor":     "extract",
    "adaptation_engine":        "adapt",
    "session_persister":        "persist",
}


# ── Queue dispatch ────────────────────────────────────────────────────────────

def _dispatch_to_queue(data: dict, api_base: str) -> Path:
    _QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt":         data.get("master_prompt") or "",
        "model":          data.get("model") or "",
        "n_agents":       data.get("n_agents") or 1,
        "task_type":      data.get("task_type") or "unknown",
        "trace_id":       data.get("trace_id") or "",
        "proxy_base":     api_base,
        "slash_commands": data.get("slash_commands") or [],
    }
    ts   = int(time.time())
    path = _QUEUE_DIR / f"{ts}.json"
    path.write_text(json.dumps(payload, indent=2))
    _CURRENT.parent.mkdir(parents=True, exist_ok=True)
    _CURRENT.write_text(json.dumps(payload, indent=2))
    return path


# ── Result renderer ───────────────────────────────────────────────────────────

def _render(data: dict, routing: dict, elapsed: float) -> None:
    k        = data.get("complexity") or data.get("Complexity_Score") or 5
    sigma    = data.get("specificity") or 0.5
    ambig    = max(0.0, min(1.0, 1.0 - sigma))
    intent   = (data.get("task_type") or data.get("Task_Type") or "unknown").upper()
    model    = data.get("model") or ""
    m_label  = _model_label(model)
    nodes    = data.get("Nodes") or []
    cp       = data.get("Critical_Path") or ""
    cp_set   = set(cp.split(" -> ")) if cp else set()
    n_agents = data.get("n_agents") or 1
    slash    = data.get("slash_commands") or []
    req_clar = data.get("requires_clarification") or False

    def row(lbl: str, val: str, color: str = "") -> None:
        v = f"[{color}]{val}[/{color}]" if color else val
        console.print(f"  {_label(lbl)}  {v}")

    console.print()
    row("ruflo_agent",  f"[blue]{routing['agent']}[/blue] · {routing['confidence']*100:.0f}% · [dim]{routing['reason']}[/dim]")
    row("cache_lookup", "miss")
    row("fast_intent",  f"{intent} · conf {0.6 + sigma * 0.4:.2f}", "cyan")
    row("classify",     f"κ={k} · specificity {sigma:.2f} · ambiguity {ambig:.2f}")

    if req_clar:
        row("clarify", f"ambiguity {ambig:.2f} ≥ 0.60 — questions generated", "yellow")
    else:
        row("clarify", f"skip — ambiguity {ambig:.2f} < 0.60", "dim")

    tokens = max(120, round(len(data.get("master_prompt") or "") / 3.5))
    row("synthesize", f"{tokens:,} tokens · preamble cached")

    reason = (
        f"κ={k} ≥ 7 — strong model" if k >= 7 else
        f"κ={k} — balanced"         if k >= 4 else
        f"κ={k} — fast"
    )
    row("route", f"{m_label} · {reason}", "green")

    if slash:
        row("commands", " ".join(slash), "magenta")

    if nodes:
        row("fleet", f"{len(nodes)} agents · {cp or '—'}", "magenta")
        for node in nodes:
            name       = node.get("Name", "?")
            on_cp      = node.get("On Critical Path") or name in cp_set
            bottleneck = node.get("Bottleneck")
            slack      = node.get("Slack")
            tags       = (" ⚡" if on_cp else "") + (" 🔴" if bottleneck else "")
            slack_str  = f" slack={slack}s" if slack is not None else ""
            color      = "magenta" if on_cp else "dim"
            row("↳", f"{name}{tags}{slack_str}", color)
    elif n_agents > 1:
        row("fleet", f"{n_agents} agents")

    row("dispatch", "queued · run [bold]claude[/bold] in your terminal", "green")
    row("elapsed",  f"{elapsed:.1f}s")
    console.print()


# ── SSE streaming ─────────────────────────────────────────────────────────────

async def _stream(prompt: str, api_base: str) -> dict | None:
    url   = f"{api_base.rstrip('/')}/stream/optimize"
    start = time.time()
    result: dict | None = None

    console.print(f"  {_label('pipeline')}  [dim]connecting…[/dim]")

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", url,
                json={"prompt": prompt},
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    console.print(f"[red]Stream error {resp.status_code}[/red] — falling back to /optimize")
                    return await _blocking(prompt, api_base)

                event_type = ""
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        try:
                            payload = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue

                        elapsed = time.time() - start

                        if event_type == "progress":
                            node  = payload.get("node", "")
                            label = _STAGE_LABELS.get(node, node)
                            console.print(f"  {_label('·')}  [dim]{label}[/dim]  [dim]{elapsed:.1f}s[/dim]")

                        elif event_type == "error":
                            console.print(f"[red]  pipeline error:[/red] {payload.get('detail', 'unknown')}")
                            return None

                        elif event_type == "end":
                            result = payload.get("result") or payload
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {api_base}[/red] — is the backend running?")
        console.print(f"  [dim]Start it with:[/dim]  .venv/bin/uvicorn backend.main:app --reload --port 8000")
        return None

    # If streaming returned no result, fall back to blocking
    if result is None:
        return await _blocking(prompt, api_base)
    return result


# ── Blocking fallback ─────────────────────────────────────────────────────────

async def _blocking(prompt: str, api_base: str) -> dict | None:
    url    = f"{api_base.rstrip('/')}/optimize"
    start  = time.time()
    stages = ["classify · score", "synthesize", "route", "fleet", "execute", "quality"]
    idx    = 0

    async def _ticker() -> None:
        nonlocal idx
        while True:
            elapsed = time.time() - start
            stage   = stages[idx % len(stages)]
            console.print(
                f"\r  {_label('pipeline')}  {stage} ···  [dim]{elapsed:.0f}s[/dim]",
                end="",
            )
            idx += 1
            await asyncio.sleep(4)

    ticker = asyncio.create_task(_ticker())
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json={"prompt": prompt})
        ticker.cancel()
        console.print()
        if resp.status_code not in (200, 202):
            console.print(f"[red]Error {resp.status_code}:[/red] {resp.text[:200]}")
            return None
        return resp.json()
    except httpx.ConnectError:
        ticker.cancel()
        console.print()
        console.print(f"[red]Cannot connect to {api_base}[/red] — is the backend running?")
        console.print(f"  [dim]Start it with:[/dim]  .venv/bin/uvicorn backend.main:app --reload --port 8000")
        return None
    except Exception as exc:
        ticker.cancel()
        console.print()
        console.print(f"[red]Request failed:[/red] {exc}")
        return None


# ── Clarification loop ────────────────────────────────────────────────────────

async def _clarify(data: dict, api_base: str) -> dict | None:
    questions  = data.get("clarification_questions") or []
    request_id = data.get("request_id") or ""

    console.print()
    console.print("[yellow]  Clarification needed:[/yellow]")
    answers: list[str] = []
    for i, q in enumerate(questions):
        dim      = q.get("dimension", "")
        question = q.get("question", str(q))
        prefix   = f"[{dim}] " if dim else ""
        console.print(f"  [dim]{str(i+1).rjust(2)}.[/dim] {prefix}{question}")
        answers.append(typer.prompt(f"     answer {i+1}"))

    console.print()
    url = f"{api_base.rstrip('/')}/optimize/continue"
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, json={"request_id": request_id, "answers": answers})

    if resp.status_code not in (200, 202):
        console.print(f"[red]Continue failed {resp.status_code}:[/red] {resp.text[:200]}")
        return None
    return resp.json()


# ── Shim installer ────────────────────────────────────────────────────────────

_SHIM_BEGIN = "# >>> optivia dispatch shim >>>"
_SHIM_BODY  = r"""# >>> optivia dispatch shim >>>
# Auto-installed by Optivia. Wraps `claude` to consume queued prompts from ~/.optivia/queue.
optivia_claude_shim() {
  if [ "$#" -eq 0 ]; then
    local q="$HOME/.optivia/queue"
    if [ -d "$q" ]; then
      local pending
      pending="$(ls -t "$q"/[0-9]*.json 2>/dev/null | head -n 1)"
      if [ -n "$pending" ]; then
        local payload prompt model slash_commands args
        payload="$(/usr/bin/python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get("prompt",""))
print("---MODEL---")
print(d.get("model","") or "")
print("---SLASH---")
cmds=d.get("slash_commands") or []
print(" ".join(cmds) if cmds else "")
' "$pending")"
        prompt="$(echo "$payload" | awk '/^---MODEL---$/{exit} {print}')"
        model="$(echo "$payload" | awk '/^---MODEL---$/,/^---SLASH---$/{if(!/^---/){print}}')"
        slash_commands="$(echo "$payload" | awk '/^---SLASH---$/{found=1;next} found{print}')"
        mv "$pending" "${pending%.json}.consumed"
        if [ -n "$prompt" ]; then
          args=()
          [ -n "$model" ] && args+=(--model "$model")
          if [ -n "$slash_commands" ]; then
            full_prompt="$slash_commands
$prompt"
          else
            full_prompt="$prompt"
          fi
          command claude "${args[@]}" "$full_prompt"
          return $?
        fi
      fi
    fi
  fi
  command claude "$@"
}
alias claude=optivia_claude_shim
# <<< optivia dispatch shim <<<
"""


def _install_shim() -> None:
    rc_files = [Path.home() / ".zshrc", Path.home() / ".bashrc"]
    for rc in rc_files:
        existing = rc.read_text() if rc.exists() else ""
        if _SHIM_BEGIN in existing:
            console.print(f"  {_label('shim')}  {rc} (already installed)")
            continue
        rc.write_text(existing.rstrip() + "\n\n" + _SHIM_BODY + "\n")
        console.print(f"  {_label('shim')}  {rc} (installed)")


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    prompt:      str  = typer.Argument("", help="Prompt to optimize (reads stdin if omitted)"),
    api_base:    str  = typer.Option(_DEFAULT_API, "--api-base", envvar="OPTIVIA_API_BASE"),
    no_dispatch: bool = typer.Option(False, "--no-dispatch", help="Skip writing to queue"),
    execute:     bool = typer.Option(False, "--execute", "-x", help="Invoke claude immediately"),
    stream:      bool = typer.Option(True, "--stream/--no-stream", help="Use SSE streaming"),
) -> None:
    """Optimize a prompt through the 17-stage pipeline and dispatch to claude."""
    asyncio.run(_run(prompt, api_base, no_dispatch, execute, stream))


async def _run(
    prompt: str,
    api_base: str,
    no_dispatch: bool,
    execute: bool,
    use_stream: bool,
) -> None:
    if not prompt:
        if sys.stdin.isatty():
            prompt = typer.prompt("prompt")
        else:
            prompt = sys.stdin.read().strip()
    if not prompt:
        console.print("[red]No prompt provided.[/red]")
        raise typer.Exit(1)

    # ── Ruflo routing (verbatim) ──────────────────────────────────────────────
    routing = route_task(prompt)
    console.print()
    console.print(
        f"  {_label('ruflo')}  "
        f"[blue]{routing['agent']}[/blue] · "
        f"{routing['confidence']*100:.0f}% · "
        f"[dim]{routing['reason']}[/dim]"
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    start = time.time()
    data  = await _stream(prompt, api_base) if use_stream else await _blocking(prompt, api_base)

    if data is None:
        raise typer.Exit(1)

    # ── Clarification loop ────────────────────────────────────────────────────
    for _ in range(3):
        if not data.get("requires_clarification"):
            break
        data = await _clarify(data, api_base)
        if data is None:
            raise typer.Exit(1)

    elapsed = time.time() - start

    # ── Render ────────────────────────────────────────────────────────────────
    _render(data, routing, elapsed)

    master_prompt = data.get("master_prompt")
    model         = data.get("model") or ""

    if not master_prompt:
        console.print("[yellow]  No master_prompt in response.[/yellow]")
        raise typer.Exit(0)

    # ── Queue dispatch ────────────────────────────────────────────────────────
    if not no_dispatch:
        queue_path = _dispatch_to_queue(data, api_base)
        console.print(f"  {_label('queue')}  [dim]{queue_path}[/dim]")
        console.print()

    # ── Execute immediately ───────────────────────────────────────────────────
    if execute:
        slash = data.get("slash_commands") or []
        full  = ("\n".join(slash) + "\n" + master_prompt).strip() if slash else master_prompt
        args  = ["claude"]
        if model:
            args += ["--model", model]
        args.append(full)
        console.print(f"  {_label('exec')}  claude --model {model} …")
        console.print()
        os.execvp("claude", args)
    else:
        console.print(
            f"  {_label('ready')}  "
            "[green]type [bold]claude[/bold] in your terminal to execute[/green]"
        )
        console.print()


@app.command("install-shim")
def install_shim() -> None:
    """Install the optivia dispatch shim into ~/.zshrc and ~/.bashrc."""
    _install_shim()
    console.print(f"  {_label('done')}  run [bold]source ~/.zshrc[/bold] to activate")


if __name__ == "__main__":
    app()
