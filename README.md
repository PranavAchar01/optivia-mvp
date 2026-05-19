# Optivia

**Pre-execution optimization layer for agentic coding CLIs.**

Optivia sits upstream of Claude Code. It takes a raw prompt, runs it through a 17-stage LangGraph pipeline — classifying, scoring, synthesizing, routing, and planning a sub-agent fleet — then dispatches an optimized mega-prompt directly into Claude Code via terminal.

```
you  →  optivia "build a REST API"  →  17-stage pipeline  →  claude
                                              ↓
                              classify · score · synthesize
                              route · fleet · schedule
                              quality · dispatch
```

---

## Features

- **17-stage LangGraph pipeline** — classification, complexity scoring, clarification, synthesis, model routing, fleet generation, CPM scheduling, quality monitoring, and experience extraction
- **Ruflo multi-agent routing** — verbatim `router.cjs` keyword-pattern routing wires every prompt to the optimal Claude Code agent type (coder, architect, reviewer, backend-dev, etc.)
- **Claude Code hooks** — `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `PreCompact`, `SubagentStop` all managed by `hook-handler.cjs`
- **Terminal-first CLI** — SSE streaming progress, clarification Q&A, queue dispatch, and optional immediate `claude` invocation
- **Shadow routing** — three routers (heuristic, RouteLLM MF, LLM judge Haiku) run in parallel; best arm selected via D-LinUCB bandit
- **Experience memory** — ExpeL-style lesson pool in PostgreSQL + `pgvector`; retrieved for every new request
- **Dispatch queue** — mega-prompt written to `~/.optivia/queue/` and consumed by a shell shim that intercepts `claude` with the correct `--model` flag and slash commands

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  optivia CLI (terminal)                                 │
│  ├─ ruflo routing  →  agent type decision               │
│  ├─ SSE stream     →  live pipeline progress            │
│  └─ dispatch       →  ~/.optivia/queue/{ts}.json        │
└────────────────────────┬────────────────────────────────┘
                         │ POST /stream/optimize
┌────────────────────────▼────────────────────────────────┐
│  FastAPI backend  (port 8000)                           │
│                                                         │
│  LangGraph pipeline — 17 nodes                         │
│  ┌──────────┐  ┌───────────┐  ┌─────────────────────┐  │
│  │ intake   │→ │ classify  │→ │ synthesize          │  │
│  │ session  │  │ score     │  │ (DSPy + caching)    │  │
│  │ exp ret. │  │ clarify?  │  └──────────┬──────────┘  │
│  └──────────┘  └───────────┘             │             │
│                                          ▼             │
│  ┌──────────────────────────────────────────────────┐  │
│  │ route · fleet gen · CPM schedule · execute       │  │
│  │ quality monitor · experience extract · persist   │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  PostgreSQL  ·  Redis  ·  pgvector  ·  Langfuse        │
└─────────────────────────────────────────────────────────┘
                         │
             ~/.optivia/queue/{ts}.json
                         │
┌────────────────────────▼────────────────────────────────┐
│  shell shim  (optivia_claude_shim in ~/.zshrc)          │
│  type `claude`  →  claude --model <model> <mega-prompt> │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL (local)
- Redis (local)
- [Claude Code CLI](https://claude.ai/code)

### 2. Clone and install

```bash
git clone https://github.com/PranavAchar01/optivia-mvp.git
cd optivia-mvp
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env   # then fill in your keys
```

Required keys:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key |
| `GEMINI_API_KEY` | Gemini API key (routing judge) |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `VOYAGE_API_KEY` | Voyage AI embeddings |
| `LANGFUSE_PUBLIC_KEY` | Observability (optional) |
| `LANGFUSE_SECRET_KEY` | Observability (optional) |

### 4. Set up the database

```bash
psql -c "CREATE DATABASE optivia;"
psql optivia -f backend/db/schema.sql
```

### 5. Start the backend

```bash
make dev
# or
.venv/bin/uvicorn backend.main:app --reload --port 8000
```

### 6. Install the shell shim

```bash
optivia install-shim
source ~/.zshrc
```

---

## CLI Usage

```bash
# Optimize a prompt — streams pipeline progress, queues for next `claude`
optivia "build a login system with Supabase auth"

# Invoke claude immediately after optimization
optivia "build a login system" --execute

# Read prompt from stdin
echo "refactor the auth module" | optivia

# Use blocking endpoint instead of SSE
optivia --no-stream "build a REST API"

# Skip writing to queue
optivia --no-dispatch "explain the codebase"

# Point at a different backend
optivia --api-base http://my-server:8000 "build X"
```

After running, type `claude` in your terminal — the shim intercepts it, picks up the queued mega-prompt, and runs:

```bash
command claude --model claude-sonnet-4-6 "<optimized mega-prompt>"
```

---

## Terminal Output

```
          ruflo  coder · 80% · Matched pattern: implement|create|build|add…

       pipeline  connecting…
              ·  intake  0.3s
              ·  cache lookup  0.4s
              ·  classify · score  18.4s
              ·  clarify?  18.4s
              ·  synthesize  31.2s
              ·  route  37.5s
              ·  fleet gen  41.0s
              ·  schedule  41.1s
              ·  execute  41.2s
              ·  quality  41.2s
              ·  persist  43.8s

     ruflo_agent  coder · 80% · Matched pattern: implement|create|build|add…
    cache_lookup  miss
     fast_intent  NEW_CODE · conf 0.90
        classify  κ=7 · specificity 0.75 · ambiguity 0.25
         clarify  skip — ambiguity 0.25 < 0.60
       synthesize  1,312 tokens · preamble cached
           route  Sonnet 4.6 · κ=7 ≥ 7 — strong model
           fleet  5 agents · Architect -> Auth Service -> DB Layer -> Tests -> Review
               ↳  Architect ⚡ slack=0s
               ↳  Auth Service ⚡ slack=0s
               ↳  DB Layer 🔴 slack=0s
               ↳  Tests  slack=15s
               ↳  Review  slack=20s
        dispatch  queued · run claude in your terminal
         elapsed  44.1s
```

---

## Ruflo Multi-Agent Routing

Optivia integrates [ruflo](https://github.com/ruvnet/ruflo)'s routing system verbatim.

**Files copied from ruflo:**

| File | Purpose |
|---|---|
| `.claude/helpers/router.cjs` | Keyword-pattern task-to-agent routing |
| `.claude/helpers/hook-handler.cjs` | Claude Code lifecycle hook dispatcher |
| `.claude/settings.json` | Hook wiring, swarm topology, MCP server config |

**Routing patterns (verbatim from `router.cjs`):**

| Pattern | Agent |
|---|---|
| `implement\|create\|build\|add` | `coder` |
| `test\|spec\|coverage` | `tester` |
| `review\|audit\|validate\|security` | `reviewer` |
| `research\|find\|search\|documentation` | `researcher` |
| `design\|architect\|structure\|plan` | `architect` |
| `api\|endpoint\|server\|backend\|database` | `backend-dev` |
| `ui\|frontend\|component\|react\|css` | `frontend-dev` |
| `deploy\|docker\|ci\|cd\|pipeline` | `devops` |

**Claude Code hook lifecycle:**

```
SessionStart      → session-restore (load patterns)
UserPromptSubmit  → route (ruflo routing decision printed to context)
PreToolUse[Bash]  → pre-bash (dangerous command guard)
PostToolUse[Edit] → post-edit (record edits, update intelligence)
SubagentStop      → post-task (train patterns on completion)
SessionEnd        → session-end (consolidate intelligence graph)
PreCompact        → compact-manual / compact-auto (guidance injection)
```

---

## Project Structure

```
optivia_mvp/
├── backend/
│   ├── core/
│   │   ├── models.py          # Pydantic domain models
│   │   ├── math_core.py       # Scorer, quality, bandit math
│   │   └── llm.py             # LLM client wrapper
│   ├── pipeline/
│   │   ├── graph.py           # LangGraph StateGraph (17 nodes)
│   │   ├── state.py           # OptiviaState TypedDict
│   │   ├── reflection.py      # Reflection utilities
│   │   └── nodes/             # One file per pipeline stage
│   ├── routing/
│   │   ├── heuristic.py       # κ-threshold router
│   │   ├── routellm.py        # Matrix-factorisation router
│   │   ├── llm_judge.py       # Haiku structured-output router
│   │   └── shadow.py          # Parallel shadow runner
│   ├── db/
│   │   ├── schema.sql         # 7-table trace contract
│   │   └── client.py          # asyncpg client
│   ├── main.py                # FastAPI app + endpoints
│   ├── proxy.py               # Anthropic proxy (mega-prompt injection)
│   ├── synthesis.py           # DSPy synthesis utilities
│   └── embeddings.py          # Voyage AI embeddings
├── cli/
│   └── main.py                # Terminal CLI (typer + rich + httpx)
├── .claude/
│   ├── settings.json          # Claude Code hooks + swarm config
│   └── helpers/
│       ├── router.cjs         # Ruflo routing (verbatim)
│       └── hook-handler.cjs   # Ruflo hook handler (verbatim)
├── tests/
├── pyproject.toml
├── Makefile
└── litellm_config.yaml
```

---

## API Reference

### `POST /optimize`

Run the full pipeline synchronously.

**Request:**
```json
{
  "prompt": "build a login system",
  "user_id": "optional",
  "workspace_id": "optional"
}
```

**Response:** `OptimizeResponse` — includes `master_prompt`, `model`, `n_agents`, fleet `Nodes`/`Edges`/`Critical_Path`, and `slash_commands`.

### `POST /stream/optimize`

Same as `/optimize` but streams `text/event-stream` SSE events:

- `event: start` — pipeline started
- `event: progress` — `{ node, message }` per completed stage
- `event: error` — pipeline error
- `event: end` — `{ result: OptimizeResponse }`

### `POST /optimize/continue`

Resume a paused clarification loop.

```json
{
  "request_id": "<uuid from initial response>",
  "answers": ["answer 1", "answer 2"]
}
```

### `GET /health`

```json
{ "status": "ok", "version": "0.1.0" }
```

---

## Development

```bash
make dev          # start backend with hot reload
make test         # run test suite
make lint         # ruff check
make typecheck    # mypy strict
```

---

## Roadmap

| Stage | Status | Description |
|---|---|---|
| 1 — Wrapper MVP | ✅ Done | Prompt optimization + Claude Code dispatch |
| 2 — Custom routing | 🔄 Planned | Train proprietary routing models on logged traces |
| 3 — V1 production | 🔄 Planned | Online learning, multi-tier execution, full caching |

---

## License

MIT
