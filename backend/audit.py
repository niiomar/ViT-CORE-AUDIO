"""
Append-only audit log for forensic chain-of-custody purposes.

Every analysis request is recorded with: file hash (SHA-256), filename,
timestamp, verdict, confidence, model version, and processing time.
The file hash means the same input file always produces a traceable record
even if uploaded under a different filename.

Unlike VIT-CORE-FORENSICS (which logs frames_analyzed), audio has no frame
concept — the field that plays a comparable "how much did the model actually
have to work with, and how much did it agree with itself" role here is
view_agreement: the cosine similarity between the L2-normalized mel and CQT
embeddings the dual-view model produces for the same clip (see model.py).
Low agreement between the two spectral views is itself a signal worth
keeping in the forensic record, the same way frame count / face quality is
for video.

Storage: SQLite at AUDIT_DB_PATH (default ./audit_log.db). For higher-volume
deployments this could be swapped for Postgres without changing the calling
code, since only `log_analysis` and `get_recent`/`get_by_hash` are used
externally.
"""

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager

from version import MODEL_VERSION

AUDIT_DB_PATH = os.getenv("AUDIT_DB_PATH", "audit_log.db")

def _init_db():
    with _connect() as conn:
        # WAL lets readers (history endpoints) proceed without blocking on
        # writers, and vice versa — needed now that analysis requests run
        # concurrently across worker threads (see main.py's asyncio.to_thread
        # usage) instead of one at a time. journal_mode is persisted in the
        # DB file itself, so this only needs to run once at startup
        # (busy_timeout, set per-connection in _connect(), handles the rest).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       REAL    NOT NULL,
                file_sha256     TEXT    NOT NULL,
                filename        TEXT    NOT NULL,
                media_type      TEXT    NOT NULL,
                verdict         TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                probability     REAL    NOT NULL,
                view_agreement  REAL    NOT NULL,
                model_version   TEXT    NOT NULL,
                processing_sec  REAL    NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_hash ON audit_log(file_sha256)")
        conn.commit()

@contextmanager
def _connect():
    conn = sqlite3.connect(AUDIT_DB_PATH)
    # busy_timeout is per-connection (unlike journal_mode, which is a
    # persisted DB property) — must be set every time.
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()

_init_db()

def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def log_analysis(file_bytes: bytes, filename: str, result: dict, model_version: str = MODEL_VERSION):
    """Record one analysis result. Best-effort — logging failures must never
    block the API response, so errors are swallowed and printed."""
    try:
        file_hash = sha256_of_bytes(file_bytes)
        with _connect() as conn:
            conn.execute(
                """INSERT INTO audit_log
                   (timestamp, file_sha256, filename, media_type, verdict,
                    confidence, probability, view_agreement, model_version, processing_sec)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    file_hash,
                    filename,
                    result.get("type", "unknown"),
                    result.get("verdict", "UNKNOWN"),
                    result.get("confidence") or 0.0,
                    result.get("probability") or 0.0,
                    result.get("view_agreement") or 0.0,
                    model_version,
                    result.get("processing_time_sec", 0.0),
                ),
            )
            conn.commit()
        return file_hash
    except Exception as e:
        print(f"[Audit] Failed to log analysis: {e}")
        return None

def get_recent(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

def get_by_hash(file_hash: str) -> list[dict]:
    """Return all past analyses for a given file hash — useful for
    'has this exact file been analysed before' checks."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE file_sha256 = ? ORDER BY id DESC", (file_hash,)
        ).fetchall()
        return [dict(r) for r in rows]
