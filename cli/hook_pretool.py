"""
Claude Code PreToolUse hook (§6.5).

Reads the tool-use event from stdin (JSON), checks it against Optivia's
guardrails, and exits 0 (allow) or non-zero (deny). The hook is wired by
`optivia install` into ~/.claude/settings.json.

Stage 1 implements a minimal safety check: block `rm -rf /` or `git push
--force` patterns that aren't in the approved plan. Stage 2 adds policy
checks against the workflow_plan stored in the active trace.
"""

from __future__ import annotations

import json
import os
import re
import sys

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/(?!\w)"),       # rm -rf /
    re.compile(r"\brm\s+-rf\s+~"),             # rm -rf ~
    re.compile(r"\bgit\s+push\s+.*--force\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bdrop\s+(table|database)\b", re.IGNORECASE),
    re.compile(r"\bsudo\s+rm\b"),
]


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0  # malformed payload — don't block

    tool_name = event.get("tool_name", "")
    if tool_name != "Bash":
        return 0

    command = (event.get("tool_input", {}) or {}).get("command", "")
    if not command:
        return 0

    for pat in _DANGEROUS_PATTERNS:
        if pat.search(command):
            sys.stderr.write(
                f"Optivia blocked dangerous bash command: {pat.pattern}\n"
                "If this was intentional, run it directly outside Claude Code.\n"
            )
            return 2  # exit code 2 → blocked per Claude Code hook protocol

    return 0


if __name__ == "__main__":
    sys.exit(main())
