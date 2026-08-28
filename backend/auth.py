"""
Simple API key authentication.

This is intentionally lightweight — a single shared secret read from the
environment, checked on every protected request via FastAPI's dependency
injection. This is NOT meant to replace proper auth (OAuth/session tokens)
for a multi-tenant production deployment, but it is sufficient to gate a
pilot/demo deployment so the inference endpoint isn't wide open to anyone
who finds the URL.

Set API_KEY in the environment (.env). If API_KEY is unset or empty,
auth is disabled entirely (useful for local dev) and a warning is logged.
"""

import hmac
import logging
import os

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY", "").strip()

if not API_KEY:
    logger.warning(
        "API_KEY is not set — the /api/v1/analyze endpoint is UNAUTHENTICATED. "
        "Set API_KEY in your environment for any non-local deployment."
    )

async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    """
    FastAPI dependency. Raises 401 if the key is missing/incorrect.

    If API_KEY is unset on the server, this is a no-op (local dev mode).
    """
    if not API_KEY:
        return  # authentication disabled

    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
