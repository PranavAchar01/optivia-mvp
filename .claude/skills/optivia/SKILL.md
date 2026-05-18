---
name: optivia
description: Optimise a vague coding task before execution. Use this skill whenever the user's prompt is ambiguous, multi-step, scope-unclear, or appears to involve more than a single trivial edit. Optivia returns an optimised master prompt, complexity score, sub-agent plan, and routing decision. Trigger especially when prompts contain phrases like "build", "refactor", "debug", "implement", "add", "design", or "figure out how to".
---

# Optivia — Pre-Execution Optimization Layer

When the user issues a coding request that is not a single-line trivial
edit, run the Optivia pipeline first. It will:

1. **Classify** the task (new_code / debug / refactor / review / explain / long / trivial / meta)
2. **Score** five dimensions: scope, ambiguity, risk, dependency, context_load
3. **Compute** the composite complexity κ (1–10) and specificity σ
4. **Ask** 1–3 clarifying questions if ambiguity is high
5. **Synthesize** an optimised master prompt with explicit scope, constraints, and acceptance criteria
6. **Route** to the right model (Haiku / Sonnet / Opus) and sub-agent count

## How to invoke

Call the MCP tool `optimize_prompt` registered under the `optivia` server:

```
optimize_prompt(prompt="<the user's raw prompt>", workspace="<repo name>")
```

If the response contains `requires_clarification: true`, present the questions
to the user in chat, gather answers, then re-call `optimize_prompt` with the
augmented prompt:

```
optimize_prompt(prompt="<original>\n\nQ: <q1>\nA: <a1>\n...", workspace="<repo>")
```

## How to use the result

The returned object contains:

- `master_prompt` — the optimised version of the user's prompt. Use this as
  your working understanding of the task instead of the raw prompt.
- `model` — Optivia's recommended Anthropic model for the task. Honour this
  signal in your planning (e.g. don't propose elaborate architectural reviews
  for a `claude-haiku-4-5` task).
- `n_agents` — recommended sub-agent count. If > 1, consider decomposing into
  parallel tasks via the Task tool.
- `slash_commands` — recommended workflow slash commands (e.g. `/plan`, `/debug`).
- `workflow_plan` — ordered steps. Treat as the default plan unless the user
  explicitly redirects.
- `complexity` (κ, 1–10) and `specificity` (σ, 0–1) — surface these to the user
  in your initial response so they understand why you picked the approach.

## When NOT to invoke

Skip Optivia when:
- The user is asking a single factual question
- The task is a single-line edit (rename, format, typo fix)
- The user has explicitly asked you to skip pre-processing

## Feedback loop

After the task is done, call `submit_feedback(trace_id=..., thumbs=1)` if the
user accepted the work, `thumbs=-1` if they rejected it. This trains Stage 2
routing.
