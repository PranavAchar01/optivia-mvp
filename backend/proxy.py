"""
Anthropic API proxy (§6.5 Surface 1).

When the user invokes `claude` via the Optivia CLI, Optivia sets
ANTHROPIC_BASE_URL=http://localhost:8000/proxy. Claude Code's outbound
calls land here. We forward them to api.anthropic.com, capture every
prompt/tool-call/response, and log them to Langfuse + Postgres so we can
build the Stage 2 training set.

We also inject Optivia's master prompt as the first system message on the
*first* call of a given OPTIVIA_REQUEST_ID, then pass through subsequent
turns untouched.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.observability import get_langfuse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/proxy", tags=["proxy"])

_ANTHROPIC_BASE = "https://api.anthropic.com"
_PROMPT_TTL = 3600  # seconds — master prompts expire after 1 hour


async def _redis():
    import redis.asyncio as aioredis
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def register_master_prompt(request_id: str, master_prompt: str) -> None:
    """Sync shim — schedules the async Redis write via the event loop."""
    if not request_id or not master_prompt:
        return
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_register_async(request_id, master_prompt))
    except RuntimeError:
        pass


async def _register_async(request_id: str, master_prompt: str) -> None:
    try:
        r = await _redis()
        async with r:
            await r.setex(f"proxy:prompt:{request_id}", _PROMPT_TTL, master_prompt)
            await r.setex(f"proxy:injected:{request_id}", _PROMPT_TTL, "0")
    except Exception as exc:
        log.warning("proxy.register_error", error=str(exc))


async def _inject_master_prompt(body: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Prepend Optivia's master prompt to the existing system message (Redis-backed)."""
    if not request_id:
        return body
    try:
        r = await _redis()
        async with r:
            already_injected = await r.get(f"proxy:injected:{request_id}")
            if already_injected == "1":
                return body
            master = await r.get(f"proxy:prompt:{request_id}")
            if not master:
                return body
            await r.setex(f"proxy:injected:{request_id}", _PROMPT_TTL, "1")
    except Exception as exc:
        log.warning("proxy.redis_lookup_error", error=str(exc))
        return body

    existing_system = body.get("system", "")
    if isinstance(existing_system, list):
        body["system"] = [
            {"type": "text", "text": master, "cache_control": {"type": "ephemeral"}},
            *existing_system,
        ]
    else:
        body["system"] = [
            {"type": "text", "text": master, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": existing_system or ""},
        ]

    log.info("proxy.master_prompt_injected", request_id=request_id, len=len(master))
    return body


def _log_to_langfuse(
    request_id: str,
    model: str,
    body: dict[str, Any],
    response_body: dict[str, Any] | None,
    elapsed_ms: int,
    is_stream: bool,
) -> None:
    lf = get_langfuse()
    if not lf or not request_id:
        return
    try:
        # langfuse v2 / v3 SDK shim
        generation = getattr(lf, "generation", None)
        if generation is None:
            return
        generation(
            trace_id=request_id,
            name="claude_code_proxy_call",
            model=model,
            input=body.get("messages") if isinstance(body, dict) else None,
            output=response_body,
            metadata={
                "is_stream": is_stream,
                "elapsed_ms": elapsed_ms,
                "system_blocks": len(body.get("system", [])) if isinstance(body.get("system"), list) else 1,
            },
        )
    except Exception as exc:
        log.warning("proxy.langfuse_log_failed", error=str(exc))


async def _handle_messages(
    request: Request,
    request_id: str,
    x_api_key: str | None,
    authorization: str | None,
    anthropic_version: str | None,
    anthropic_beta: str | None,
) -> Response:
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}

    body = await _inject_master_prompt(body, request_id)
    is_stream = bool(body.get("stream"))

    # Re-serialise the (possibly mutated) body
    forward_bytes = json.dumps(body).encode("utf-8") if body else body_bytes

    headers = {
        "x-api-key": x_api_key or settings.anthropic_api_key,
        "anthropic-version": anthropic_version or "2023-06-01",
        "content-type": "application/json",
    }
    if anthropic_beta:
        headers["anthropic-beta"] = anthropic_beta
    if authorization:
        headers["authorization"] = authorization

    target_url = f"{_ANTHROPIC_BASE}/v1/messages"
    t0 = time.monotonic()

    if is_stream:
        async def stream_proxy():
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST", target_url, headers=headers, content=forward_bytes
                ) as upstream:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk

        _log_to_langfuse(
            request_id=request_id,
            model=body.get("model", "?"),
            body=body,
            response_body={"status": "streaming"},
            elapsed_ms=0,
            is_stream=True,
        )
        return StreamingResponse(stream_proxy(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(target_url, headers=headers, content=forward_bytes)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    try:
        resp_json = resp.json()
    except Exception:
        resp_json = None

    _log_to_langfuse(
        request_id=request_id,
        model=body.get("model", "?"),
        body=body,
        response_body=resp_json,
        elapsed_ms=elapsed_ms,
        is_stream=False,
    )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# Path-scoped variant: ANTHROPIC_BASE_URL=http://localhost:8000/proxy/req/<id>
# Claude Code will then POST to /proxy/req/<id>/v1/messages
@router.post("/req/{request_id}/v1/messages")
async def messages_proxy_scoped(
    request_id: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="x-api-key"),
    authorization: str | None = Header(None),
    anthropic_version: str | None = Header(None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(None, alias="anthropic-beta"),
) -> Response:
    return await _handle_messages(
        request, request_id, x_api_key, authorization, anthropic_version, anthropic_beta
    )


# Header-based variant: clients that can set x-optivia-request-id
@router.post("/v1/messages")
async def messages_proxy_default(
    request: Request,
    x_api_key: str | None = Header(None, alias="x-api-key"),
    authorization: str | None = Header(None),
    anthropic_version: str | None = Header(None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(None, alias="anthropic-beta"),
    x_optivia_request_id: str | None = Header(None, alias="x-optivia-request-id"),
) -> Response:
    request_id = x_optivia_request_id or request.query_params.get("optivia_request_id", "")
    return await _handle_messages(
        request, request_id, x_api_key, authorization, anthropic_version, anthropic_beta
    )


@router.get("/health")
async def proxy_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "upstream": _ANTHROPIC_BASE,
    }
