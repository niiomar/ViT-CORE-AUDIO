# Stage 1: build the frontend
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
# VITE_API_KEY is baked into the bundle at build time
ARG VITE_API_KEY
ENV VITE_API_KEY=${VITE_API_KEY}
RUN npm run build

# Stage 2: backend runtime
#
# Pinned to bookworm explicitly, matching VIT-CORE-FORENSICS' Dockerfile —
# the floating python:3.11-slim tag can move to a newer Debian release out
# from under an apt package version pin without warning.
FROM python:3.11-slim-bookworm AS runtime

# soundfile ships its own bundled libsndfile in the wheel, so no system
# package is required for wav/flac/ogg. ffmpeg is what actually matters
# here: librosa falls back to it (via audioread) for mp3/m4a/aac/wma, and
# opencv-python-headless (used only for colorizing the mel/CQT view images
# returned to the frontend, never for the model's own input) still needs
# the same base X11/GL libs VIT-CORE-FORENSICS pins for the same reason.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0=2.74.6-2+deb12u9 \
    libsm6=2:1.2.3-1 \
    libxext6=2:1.3.4-1+b1 \
    libxrender1=1:0.9.10-1.1 \
    libgl1=1.6.0-1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# vitcore_audio is the shared package (model architecture + audio
# preprocessing) also used by train.py/evaluate.py at the repo root —
# installed once here so backend/requirements.txt's `-e ..` (see that
# file) resolves against it below, instead of vendoring a second copy.
COPY pyproject.toml ./
COPY vitcore_audio/ ./vitcore_audio/

WORKDIR /app/backend

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

# vite.config.js sets outDir to ../backend/static (relative to /app/frontend
# in the build stage), not Vite's default ./dist — copy from where the
# build actually lands.
COPY --from=frontend-build /app/backend/static ./static

# Model weights and audit DB are expected to be mounted as volumes.
# Sibling to /app/backend (not nested under it) — matches
# docker-compose.yml's ./backend/weights:/app/weights and
# ./backend/data:/app/data mount targets.
RUN mkdir -p /app/weights /app/data

# Container-native defaults matching the volume mount points above — the
# bare-metal defaults in .env.example (relative paths, meant for running
# from backend/ directly) resolve wrong here. Still overridable via
# .env/docker-compose environment if needed.
ENV MODEL_WEIGHTS_PATH=/app/weights/vitcore_audio_best.pth
ENV AUDIT_DB_PATH=/app/data/audit_log.db

# Run as a non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/sh --no-create-home appuser \
    && chown -R appuser:appgroup /app/data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
