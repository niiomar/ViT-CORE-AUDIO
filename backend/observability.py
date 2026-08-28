"""
Request correlation, structured logging, and Prometheus metrics.

Kept as a single small module rather than pulling in a framework — this
project's own convention (see auth.py, ratelimit.py) is lightweight, native
implementations over heavy dependencies where reasonable. prometheus_client
is the one exception: hand-rolling histogram bucketing correctly is not
worth it, and the package itself is a small, pure-Python, well-maintained
dependency with no transitive bloat.
"""

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

REQUEST_COUNT = Counter("vitcoreaudio_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("vitcoreaudio_request_latency_seconds", "Request latency in seconds", ["method", "path"])
BATCH_SIZE = Histogram(
    "vitcoreaudio_batch_size",
    "Number of files per /api/v1/analyze/batch request",
    buckets=(1, 2, 5, 10, 20, 50),
)


class _RequestIdFilter(logging.Filter):
    """Injects the current request's ID into every log record so log lines
    from the same request can be grepped together, even across the
    worker-thread boundary introduced by asyncio.to_thread."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    """Text logs by default (matches this project's existing console
    output); set LOG_FORMAT=json for structured logs suitable for a log
    aggregator in a real deployment."""
    json_logs = os.getenv("LOG_FORMAT", "text").strip().lower() == "json"

    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [req:%(request_id)s] %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]


def get_request_id() -> str:
    return _request_id_ctx.get()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID (or reuses an inbound X-Request-ID from a
    reverse proxy), times the request, records Prometheus metrics, and
    echoes the ID back so a client/proxy can correlate a response with
    server-side logs."""

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = _request_id_ctx.set(request_id)
        start = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            _request_id_ctx.reset(token)

        duration = time.monotonic() - start
        # Route *template* (e.g. "/api/v1/history/{file_hash}"), not the
        # resolved path — using the raw path would give every distinct file
        # hash, static asset filename, or 404-probe its own Prometheus label,
        # growing cardinality unboundedly. Unmatched routes (404s) share one
        # "unmatched" bucket for the same reason.
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(duration)
        response.headers["X-Request-ID"] = request_id
        return response


def metrics_response():
    return generate_latest(), CONTENT_TYPE_LATEST
