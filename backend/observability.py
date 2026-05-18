"""Langfuse + OpenTelemetry wiring for the Optivia pipeline (§9.1)."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

import structlog
from langfuse import Langfuse

from backend.config import settings

log = structlog.get_logger(__name__)

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    global _langfuse
    if _langfuse is None and settings.langfuse_public_key:
        try:
            _langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception as exc:
            log.warning("langfuse.init_failed", error=str(exc))
    return _langfuse


def trace_node(node_name: str):
    """Decorator that wraps a LangGraph node with a Langfuse span."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            lf = get_langfuse()
            trace_id = state.get("trace_id") or state.get("request_id", "")
            span = None

            if lf and trace_id:
                try:
                    span = lf.span(
                        trace_id=trace_id,
                        name=node_name,
                        input={"request_id": state.get("request_id", "")},
                    )
                except Exception:
                    pass

            t0 = time.monotonic()
            try:
                result = await fn(state)
                elapsed_ms = int((time.monotonic() - t0) * 1000)

                if span:
                    try:
                        span.end(
                            output={
                                "elapsed_ms": elapsed_ms,
                                "error": result.get("error"),
                            }
                        )
                    except Exception:
                        pass

                return result
            except Exception as exc:
                if span:
                    try:
                        span.end(output={"error": str(exc)}, level="ERROR")
                    except Exception:
                        pass
                raise

        return wrapper
    return decorator


def emit_trace_score(trace_id: str, quality_score: float, cost_usd: float = 0.0) -> None:
    """Emit score-level telemetry to Langfuse (§9.1 score level)."""
    lf = get_langfuse()
    if not lf or not trace_id:
        return
    try:
        # SDK ≥2.36 uses score_create; older versions used score
        fn = getattr(lf, "score_create", None) or getattr(lf, "score", None)
        if fn:
            fn(
                trace_id=trace_id,
                name="quality_scalar",
                value=quality_score,
                comment=f"cost_usd={cost_usd:.4f}",
            )
    except Exception as exc:
        log.warning("langfuse.score_failed", error=str(exc))
