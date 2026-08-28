"""
FastAPI app: routes, CORS, rate limiting, request/metrics middleware, and
lifespan-managed model loading. The actual inference pipeline lives in
model.py; this module handles HTTP concerns and per-request orchestration
(tempfile staging, audit logging) — the audio analogue of FORENSICS'
frame-extraction orchestration in its own main.py.
"""

from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
loaded = load_dotenv(dotenv_path=env_path)
print(f"[Config] .env loaded: {loaded} (path: {env_path})")

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

import audit
from auth import verify_api_key
from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from model import analyze_audio, get_model, get_status
from observability import (
    BATCH_SIZE,
    ObservabilityMiddleware,
    configure_logging,
    metrics_response,
)
from ratelimit import enforce_rate_limit
from version import MODEL_VERSION

configure_logging()
logger = logging.getLogger(__name__)

# soundfile (via librosa) natively reads wav/flac/ogg; mp3/m4a/aac/wma go
# through librosa's audioread fallback, which needs ffmpeg on PATH in the
# deployment environment — see README's Installation notes.
SUPPORTED_EXTENSIONS = ('.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aac', '.wma')

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing neural weights...")
    get_model()
    yield
    logger.info("Shutting down.")

app = FastAPI(title="ViT-CORE-Audio API", version=MODEL_VERSION, lifespan=lifespan)

# Security: Explicit origins for local Vite development and cross-port traffic
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Merge any additional origins from .env
env_cors = os.getenv("CORS_ORIGINS", "")
if env_cors:
    CORS_ORIGINS.extend([o.strip() for o in env_cors.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows OPTIONS pre-flight checks required by browsers
    allow_headers=["*"],  # Allows custom headers like X-API-KEY
)

# Assigns/propagates a request ID (X-Request-ID) for log correlation and
# records Prometheus metrics for every request. Added last so it's the
# outermost middleware and its timing captures CORS handling too.
app.add_middleware(ObservabilityMiddleware)


# Staging: audio decoders (librosa/soundfile/audioread) need a real file
# path, not an in-memory buffer, for every format this service accepts —
# so the upload is written to a tempfile first, exactly the way FORENSICS'
# extract_frames_to_pil stages video before cv2.VideoCapture. This does
# purely blocking work (tempfile I/O, librosa decode, the forward pass),
# so it runs off the event loop via asyncio.to_thread — see _run_analysis
# below — which is what lets analyze_batch process multiple files
# concurrently instead of one at a time.
def _run_analysis_sync(filename: str | None, content: bytes, explain: bool) -> dict:
    start_time = time.time()
    filename_lower = (filename or "").lower()

    if not filename_lower.endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {filename}")
    assert filename is not None  # guaranteed by the check above: "".endswith(...) is always False

    file_suffix = Path(filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = analyze_audio(tmp_path, generate_visuals=explain)
    except Exception as e:
        logger.error(f"Analysis pipeline error for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not analyze {filename}: {e}") from e
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    result["filename"] = filename
    result["type"] = filename_lower.rsplit(".", 1)[-1]
    result["file_size_bytes"] = len(content)
    result["processing_time_sec"] = round(time.time() - start_time, 2)

    file_hash = audit.log_analysis(content, filename, result, model_version=MODEL_VERSION)
    result["file_sha256"] = file_hash

    return result

async def _run_analysis(filename: str | None, content: bytes, explain: bool) -> dict:
    """Async wrapper: offloads the blocking analysis pipeline to a worker
    thread so it doesn't stall the event loop for other in-flight requests."""
    return await asyncio.to_thread(_run_analysis_sync, filename, content, explain)

# Routes
@app.post("/api/v1/analyze", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
async def analyze_media(file: UploadFile = File(...), explain: bool = Query(default=True)):
    """explain=True (default) also returns the colorized mel/CQT spectrogram
    views for the frontend's Mel/CQT tabs. Batch requests default it off —
    see analyze_batch."""
    logger.info(f"Analyzing asset: {file.filename}")
    content = await file.read()
    try:
        return await _run_analysis(file.filename, content, explain)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis pipeline error: {e!s}")
        raise HTTPException(status_code=500, detail=str(e)) from e

# Batch files are processed concurrently, bounded by BATCH_CONCURRENCY so a
# large batch doesn't spawn many simultaneous model forward passes and blow
# out GPU/CPU memory. The actual model inference is further serialized
# inside model.py (see _inference_lock) — the concurrency win here comes
# from overlapping I/O-bound work (tempfile staging, librosa decode, audit
# writes) across files while inference queues behind the lock.
BATCH_CONCURRENCY = int(os.getenv("BATCH_CONCURRENCY", "4"))

@app.post("/api/v1/analyze/batch", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
async def analyze_batch(files: list[UploadFile] = File(...), explain: bool = Query(default=False)):
    """
    Analyze multiple files in one request. Each file is processed
    independently; a failure on one file does not abort the others —
    its entry in the response will contain an "error" field instead
    of the usual result fields.

    Spectrogram visuals default to OFF for batch requests since they're
    not rendered anywhere in a batch summary and add response size for
    no benefit — a batch is typically a screening pass, not a deep-dive
    on a single file.
    """

    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Batch size limited to 50 files per request.")

    logger.info(f"Batch analyzing {len(files)} assets")
    BATCH_SIZE.observe(len(files))

    semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def process_one(f: UploadFile) -> dict:
        content = await f.read()
        async with semaphore:
            try:
                return await _run_analysis(f.filename, content, explain)
            except HTTPException as e:
                return {"filename": f.filename, "error": e.detail}
            except Exception as e:
                logger.error(f"Batch item error ({f.filename}): {e}")
                return {"filename": f.filename, "error": str(e)}

    results = await asyncio.gather(*(process_one(f) for f in files))

    summary = {
        "total": len(results),
        "spoof": sum(1 for r in results if r.get("verdict") == "SPOOF"),
        "bonafide": sum(1 for r in results if r.get("verdict") == "BONAFIDE"),
        "errors": sum(1 for r in results if "error" in r),
    }
    return {"summary": summary, "results": results}

@app.get("/api/v1/history", dependencies=[Depends(verify_api_key)])
async def history(limit: int = Query(default=50, le=200)):
    """Return recent audit log entries (chain-of-custody view)."""
    return {"entries": audit.get_recent(limit)}

@app.get("/api/v1/history/{file_hash}", dependencies=[Depends(verify_api_key)])
async def history_by_hash(file_hash: str):
    """Return all past analyses for a given SHA-256 file hash."""
    entries = audit.get_by_hash(file_hash)
    if not entries:
        raise HTTPException(status_code=404, detail="No records for this file hash.")
    return {"entries": entries}

@app.get("/health")
async def health():
    status = get_status()
    body = {"status": "ok" if status["model_loaded"] else "degraded", "version": MODEL_VERSION, **status}
    if not status["model_loaded"]:
        # A process that's up but never actually loaded the model (missing
        # weights, a load-time exception swallowed elsewhere) isn't really
        # healthy — it'll 500 on the first real request. Returning non-2xx
        # here is what makes the Docker/compose HEALTHCHECK meaningful
        # instead of just checking the port is open.
        raise HTTPException(status_code=503, detail=body)
    return body

@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint. Deliberately unauthenticated (like
    /health) to match standard scraper setups — restrict network access to
    it at the reverse-proxy/firewall level in any exposed deployment,
    same as you would for any other internal metrics endpoint."""
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)

# Serves the Vite production build (npm run build outputs here) so the
# whole app is a single FastAPI process in production.
_static = Path(__file__).parent / "static"

if _static.exists():
    # Mount the /assets folder so JS and CSS load correctly
    _assets = _static / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    # Serve the main HTML file at the root URL
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(_static / "index.html"))
else:
    logger.warning(
        f"Frontend build directory not found at {_static}. "
        "Check your Vite configuration."
    )
