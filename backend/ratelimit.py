"""
Lightweight in-memory sliding-window rate limiter for the inference endpoints.

Keyed by API key when auth is enabled (so each credentialed client gets its
own budget), falling back to client IP otherwise. Intentionally
dependency-free to match the rest of this project's "native" approach — for
a multi-worker or multi-instance deployment this state is per-process, so
pair it with a reverse-proxy-level limiter (Nginx, Caddy) or swap this for a
Redis-backed limiter if you scale beyond a single worker.

Configured via env vars:
  RATE_LIMIT_REQUESTS        max requests per window (default 20)
  RATE_LIMIT_WINDOW_SECONDS  window size in seconds (default 60)

Set RATE_LIMIT_REQUESTS=0 to disable.
"""

import os
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

_hits: dict[str, deque] = defaultdict(deque)


def _client_key(request: Request) -> str:
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


async def enforce_rate_limit(request: Request):
    """FastAPI dependency. Raises 429 once the caller exceeds its budget
    for the current sliding window."""
    if RATE_LIMIT_REQUESTS <= 0:
        return  # disabled

    key = _client_key(request)
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    hits = _hits[key]
    while hits and hits[0] < window_start:
        hits.popleft()

    if len(hits) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )

    hits.append(now)
