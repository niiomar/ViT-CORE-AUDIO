"""
HTTP-layer tests for main.py's FastAPI endpoints: auth, rate limiting,
response shapes, and batch/history routing. Complements test_smoke.py,
which tests the inference pipeline directly without going through HTTP.
Uses a real (untrained, CI has no checkpoint) model on CPU throughout.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Must be set before importing main (and everything it imports) — auth.py,
# ratelimit.py, and audit.py all read their config from env vars once, at
# module import time.
os.environ["MODEL_WEIGHTS_PATH"] = "nonexistent.pth"
os.environ["API_KEY"] = "test-api-key"
os.environ["RATE_LIMIT_REQUESTS"] = "3"
os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "60"
os.environ["AUDIT_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="vitcoreaudio_test_audit_"), "audit_log.db")

import main
import ratelimit
from fastapi.testclient import TestClient

HEADERS = {"X-API-KEY": "test-api-key"}


def _synthetic_wav_bytes(seconds: float = 4.0, sr: int = 16000, freq: float = 220.0) -> bytes:
    import io
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c
    ratelimit._hits.clear()  # don't let one test's requests count against the next


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metrics_endpoint_is_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"vitcoreaudio_requests_total" in r.content


def test_analyze_requires_api_key(client):
    r = client.post("/api/v1/analyze", files={"file": ("x.wav", _synthetic_wav_bytes(), "audio/wav")})
    assert r.status_code == 401


def test_analyze_rejects_unsupported_extension(client):
    r = client.post(
        "/api/v1/analyze",
        files={"file": ("evidence.txt", b"not audio", "text/plain")},
        headers=HEADERS,
    )
    assert r.status_code == 400


def test_analyze_returns_full_result_and_logs_audit(client):
    r = client.post(
        "/api/v1/analyze?explain=true",
        files={"file": ("clip.wav", _synthetic_wav_bytes(), "audio/wav")},
        headers=HEADERS,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["verdict"] in ("BONAFIDE", "SPOOF")
    assert 0.0 <= data["probability"] <= 1.0
    assert -1.0 <= data["view_agreement"] <= 1.0
    assert data["file_sha256"] is not None
    assert data["file_size_bytes"] == len(_synthetic_wav_bytes())
    assert data["visuals"]["mel"] != ""
    assert data["visuals"]["cqt"] != ""


def test_analyze_omits_visuals_when_explain_false(client):
    r = client.post(
        "/api/v1/analyze?explain=false",
        files={"file": ("clip.wav", _synthetic_wav_bytes(), "audio/wav")},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["visuals"] == {"mel": "", "cqt": ""}


def test_batch_rejects_over_50_files(client):
    files = [("files", (f"{i}.wav", _synthetic_wav_bytes(), "audio/wav")) for i in range(51)]
    r = client.post("/api/v1/analyze/batch", files=files, headers=HEADERS)
    assert r.status_code == 400


def test_batch_processes_multiple_files(client):
    files = [("files", (f"{i}.wav", _synthetic_wav_bytes(), "audio/wav")) for i in range(3)]
    r = client.post("/api/v1/analyze/batch", files=files, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == 3
    assert len(body["results"]) == 3


def test_rate_limit_returns_429_after_threshold(client):
    # RATE_LIMIT_REQUESTS=3 for this test session (set above)
    codes = []
    for _ in range(5):
        r = client.post(
            "/api/v1/analyze?explain=false",
            files={"file": ("clip.wav", _synthetic_wav_bytes(), "audio/wav")},
            headers=HEADERS,
        )
        codes.append(r.status_code)
    assert codes[:3] == [200, 200, 200]
    assert 429 in codes[3:]


def test_history_requires_api_key(client):
    r = client.get("/api/v1/history")
    assert r.status_code == 401


def test_history_returns_entries_after_analysis(client):
    client.post(
        "/api/v1/analyze?explain=false",
        files={"file": ("clip.wav", _synthetic_wav_bytes(), "audio/wav")},
        headers=HEADERS,
    )
    r = client.get("/api/v1/history", headers=HEADERS)
    assert r.status_code == 200
    assert len(r.json()["entries"]) >= 1


def test_history_by_unknown_hash_returns_404(client):
    r = client.get("/api/v1/history/" + "0" * 64, headers=HEADERS)
    assert r.status_code == 404


def test_metrics_uses_route_template_not_raw_path(client):
    """Regression guard: the observability middleware must label metrics
    with the route template, not the raw resolved URL — otherwise every
    distinct file hash grows Prometheus cardinality unboundedly."""
    client.get("/api/v1/history/" + "a" * 64, headers=HEADERS)
    client.get("/api/v1/history/" + "b" * 64, headers=HEADERS)
    body = client.get("/metrics").text
    assert 'path="/api/v1/history/{file_hash}"' in body
    assert "a" * 64 not in body
    assert "b" * 64 not in body
