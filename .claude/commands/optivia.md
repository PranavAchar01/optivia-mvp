---
description: Optimise a coding prompt with Optivia before executing
argument-hint: <the messy prompt to optimise>
---

Invoke the Optivia MCP server's `optimize_prompt` tool with the user's prompt below. Use the returned `master_prompt`, `workflow_plan`, and `model` recommendation to guide your subsequent work. Surface `complexity` (κ) and `specificity` (σ) to the user before starting.

If `requires_clarification` is true, present the questions to the user, gather answers, then re-call the tool with the augmented prompt.

User's prompt: $ARGUMENTS
