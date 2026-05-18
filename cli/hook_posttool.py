"""
Claude Code PostToolUse hook (§6.5).

Reads the tool-use event from stdin (JSON), forwards an observation event
to the Optivia backend so the Quality Monitor can update Q_t.
"""

from __future__ import annotations

import json
import os
import sys
from urllib import request

_API_BASE = os.environ.get("OPTIVIA_API_BASE", "http://localhost:8000")
_TRACE_ID = os.environ.get("OPTIVIA_TRACE_ID", "")
_REQUEST_ID = os.environ.get("OPTIVIA_REQUEST_ID", "")


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0

    payload = {
        "request_id": _REQUEST_ID,
        "trace_id": _TRACE_ID,
        "tool_name": event.get("tool_name", ""),
        "tool_input": event.get("tool_input", {}),
        "tool_response": event.get("tool_response", {}),
    }

    try:
        req = request.Request(
            f"{_API_BASE}/internal/observe",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=2) as _:
            pass
    except Exception:
        # Hook failures must not interrupt Claude Code — swallow.
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
