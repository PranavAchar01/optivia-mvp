"""
voyage-code-3 embeddings client (§4.5, §5.7).

Used for both Tier 1 semantic cache lookups and the raw_prompt_emb column
that powers Stage 2 kNN retrieval over historical successful master prompts.
"""

from __future__ import annotations

import asyncio
import math
from typing import Optional

import httpx
import structlog

from backend.config import settings

log = structlog.get_logger(__name__)

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
_MODEL = "voyage-code-3"
_DIM = 1024


def _zero_vector() -> list[float]:
    return [0.0] * _DIM


async def embed(text: str, input_type: str = "document") -> list[float]:
    """
    Returns a 1024-d voyage-code-3 embedding. Falls back to a zero vector
    on any failure — callers should treat zero vectors as cache misses.
    """
    if not settings.voyage_api_key or not text:
        return _zero_vector()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                _VOYAGE_URL,
                headers={
                    "Authorization": f"Bearer {settings.voyage_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": [text[:8000]],   # voyage-code-3 has an 8K context
                    "model": _MODEL,
                    "input_type": input_type,  # "document" or "query"
                },
            )
            if r.status_code != 200:
                log.warning("voyage.error", status=r.status_code, body=r.text[:200])
                return _zero_vector()
            data = r.json()
            return data["data"][0]["embedding"]
    except Exception as exc:
        log.warning("voyage.exception", error=str(exc))
        return _zero_vector()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
