# ViT-CORE-Audio — Service

FastAPI + Vite microservice (`backend/` + `frontend/`) wrapping this repo's dual-view
(mel-spectrogram + Constant-Q Transform) anti-spoofing model, built to run alongside
[VIT-CORE-FORENSICS](https://github.com/niiomar/VIT-CORE-FORENSICS) and
[c2pa-veritas](https://github.com/niiomar/c2pa-veritas) as a third microservice for
[veritas-nexus](https://github.com/niiomar/veritas-nexus). It mirrors those two services'
conventions deliberately — same auth, rate limiting, audit logging, observability, and
checkpoint-loading pattern — so it drops into the existing deployment shape without
introducing a new one.

The model architecture, training loop, and dataset handling live in the `vitcore_audio/`
package at the repo root — see the [main README](README.md) for that side. `backend/model.py`
imports `ViTCoreAudio` and `load_dual_views` directly from `vitcore_audio` rather than
vendoring copies, so inference always matches exactly what a checkpoint trained here was
evaluated against — no separate copy to keep in sync.

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Integrating with veritas-nexus](#integrating-with-veritas-nexus)
- [Security & Deployment Notes](#security--deployment-notes)

## Architecture

```
Upload (wav/flac/mp3/m4a/ogg/aac/wma)
        │
        ▼
  tempfile staging  (main.py)
        │
        ▼
  load_dual_views()  (audio_preprocessing.py, vendored from the training repo)
   ├── mel-spectrogram view  (224×224×3)
   └── CQT view              (224×224×3)
        │
        ▼
  shared ViT-S/16 encoder  (model.py::ViTCoreAudio)
   ├── f1_norm (mel embedding, L2-normalized)
   └── f2_norm (CQT embedding, L2-normalized)
        │
        ▼
  fused logits → softmax → spoof probability
  cosine_similarity(f1_norm, f2_norm) → view_agreement
        │
        ▼
  verdict (BONAFIDE / SPOOF) + confidence + view_agreement
        │
        ▼
  audit.py → SQLite chain-of-custody log
```

`view_agreement` is this service's analogue of VIT-CORE-FORENSICS' frame/face-quality
signal: audio has no frame concept, but the dual-view architecture already produces two
independent embeddings of the same clip, so their cosine similarity is a free, meaningful
"how much did the model agree with itself" signal — logged alongside every verdict and
surfaced in the frontend as a warning when it drops below a threshold.

## Project Structure

```
vitcore_audio/                 Shared package (repo root) — see the main README, not vendored here
backend/
  main.py                  FastAPI app: routes, CORS, rate limiting, observability, lifespan
  model.py                 Checkpoint loading + analyze_audio() — imports ViTCoreAudio/load_dual_views
                            from vitcore_audio, doesn't redefine them
  auth.py                  Shared-secret X-API-KEY auth
  ratelimit.py              In-memory sliding-window rate limiter
  audit.py                  SQLite chain-of-custody log
  observability.py          Request-ID logging + Prometheus metrics
  version.py                 MODEL_VERSION string
  tests/                     test_smoke.py (pipeline), test_api.py (HTTP layer)
  weights/                   Mount point for the trained checkpoint (not committed)
frontend/
  src/app.js                 Application logic
  src/components/            sidebar.js, workspace.js, history.js
  src/utils/                 api.js, report.js (PDF export)
  src/styles.css              Dark forensic theme, shared visual language with FORENSICS
Dockerfile                    Two-stage build: frontend (Vite) → installs vitcore_audio → backend runtime
docker-compose.yml             Host port 8003 (VIT-CORE=8001, c2pa-veritas=8002)
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node 18+
- `ffmpeg` on PATH (only needed for mp3/m4a/aac/wma — wav/flac/ogg decode natively via
  soundfile, no extra dependency)

### 1. Back end

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set API_KEY, and MODEL_WEIGHTS_PATH once you have a real checkpoint
# (see below — the service runs on untrained weights with a logged warning if you don't)

mkdir -p weights
# Copy your trained checkpoint (e.g. vitcore_audio_best.pth from the training repo's
# train.py --checkpoint_dir output) into weights/
```

### 2. Front end

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_KEY to match backend/.env's API_KEY
npm run build           # outputs into ../backend/static
```

### 3. Run

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` — the FastAPI process serves both the API and the built
frontend from one process.

For frontend development with hot reload instead: `npm run dev` (from `frontend/`) proxies
`/api` to `http://127.0.0.1:8003` per `vite.config.js` — run the backend on port 8003
locally to match, or edit the proxy target.

### Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest -v
```

`test_smoke.py` exercises the full dual-view pipeline directly (no checkpoint needed —
CI runs on untrained weights, same as VIT-CORE-FORENSICS' convention). `test_api.py` drives
the actual FastAPI app through `TestClient` with lifespan, covering auth, rate limiting,
batch handling, and audit history.

## Configuration

### `backend/.env`

See `backend/.env.example` for the full list with explanations. The load-bearing ones:

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | *(unset — auth disabled)* | Shared secret for `X-API-KEY` |
| `MODEL_WEIGHTS_PATH` | `weights/vitcore_audio_best.pth` | Trained checkpoint path |
| `MODEL_WEIGHTS_SHA256` | *(unset)* | Pin the checkpoint hash; refuses to start on mismatch |
| `ALLOW_UNTRUSTED_CHECKPOINT` | `false` | Opt-in fallback to unsafe pickle deserialization |
| `RATE_LIMIT_REQUESTS` / `_WINDOW_SECONDS` | `20` / `60` | Sliding-window rate limit |
| `AUDIT_DB_PATH` | `audit_log.db` | SQLite chain-of-custody log |

### `frontend/.env`

`VITE_API_KEY` — baked into the JS bundle at build time, must match `backend/.env`'s
`API_KEY`.

## Docker Deployment

```bash
cp backend/.env.example backend/.env    # edit before building
cp frontend/.env.example frontend/.env
docker compose up --build -d
```

Bound to `127.0.0.1:8003` by default (see `docker-compose.yml`'s comments — matches
`VIT_CORE_URL`'s 8001 and `C2PA_URL`'s 8002 convention in veritas-nexus's `worker.py`).
Mount your trained checkpoint at `backend/weights/vitcore_audio_best.pth` — it's bind-mounted
read-only into the container.

## API Reference

### `POST /api/v1/analyze`

Multipart upload, field name `file`. Query param `explain` (default `true`) controls whether
the mel/CQT view images are rendered and returned — turn it off for a raw-score-only path.

```json
{
  "probability": 0.4323,
  "verdict": "BONAFIDE",
  "confidence": 56.8,
  "view_agreement": 0.967,
  "is_low_confidence": true,
  "processing_time_sec": 1.64,
  "visuals": { "mel": "<base64 jpeg>", "cqt": "<base64 jpeg>" },
  "filename": "clip.wav",
  "type": "wav",
  "file_size_bytes": 128044,
  "file_sha256": "136b6d..."
}
```

### `POST /api/v1/analyze/batch`

Same shape, `files` (plural) field, up to 50 files. `explain` defaults to `false`. Returns
`{"summary": {"total", "bonafide", "spoof", "errors"}, "results": [...]}`.

### `GET /api/v1/history?limit=50` / `GET /api/v1/history/{file_sha256}`

Chain-of-custody read access to the audit log.

### `GET /health` / `GET /metrics`

Unauthenticated. `/health` returns 503 if the model never loaded; `/metrics` is
Prometheus-format (`vitcoreaudio_*` series).

## Integrating with veritas-nexus

veritas-nexus's `api/worker.py` currently **hard-blocks all audio files** at its
`is_valid_visual_media` gatekeeper (extension, MIME type, and magic-byte checks all reject
audio explicitly) — that block predates this service and will need to be relaxed for audio
uploads to reach it at all.

The existing `call_vit_core_microservice`/`verify_c2pa_provenance` functions in `worker.py`
are the pattern to follow: read `AUDIO_URL`/`AUDIO_API_KEY` from the environment (defaulting
to `http://host.docker.internal:8003/api/v1/analyze`, matching this service's
`docker-compose.yml` port), POST the file, and read `response.json()["probability"]` — this
service's response shape already matches that exact contract. That integration work hasn't
been done yet; this repo is the standalone service only.

## Security & Deployment Notes

- `API_KEY` and rate limiting are pilot/demo-grade, not multi-tenant auth — see `auth.py`'s
  and `ratelimit.py`'s own docstrings.
- The checkpoint-loading path in `model.py` refuses to fall back to unsafe pickle
  deserialization unless `ALLOW_UNTRUSTED_CHECKPOINT` is explicitly set — same reasoning as
  VIT-CORE-FORENSICS' `model.py`.
- `/metrics` is unauthenticated by convention (matching standard Prometheus scraper setups)
  — restrict it at the reverse-proxy/firewall level in any exposed deployment.
