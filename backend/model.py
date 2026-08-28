"""
Inference-serving glue for ViT-CORE-Audio: checkpoint loading, the
tempfile-to-verdict pipeline, and mel/CQT view visualization.

The model architecture itself (ViTCoreAudio) and the preprocessing
transform (load_dual_views) are NOT defined here — they're imported from
the vitcore_audio package (repo root), the same package train.py and
evaluate.py use. This file used to vendor its own copies of both; that
created a real drift risk (inference silently diverging from what a
checkpoint was actually trained against) and is exactly what moving the
shared code into an installable package was meant to eliminate.

analyze_audio is the core entry point — it runs the eval-time transform
(load_dual_views -> ToTensor, no augmentation, matching datasets.py's
train=False path exactly) through the shared ViT-S/16 encoder and returns a
verdict, confidence, and the mel/CQT view-agreement signal used in place of
FORENSICS' frame/face-quality fields.
"""

import base64
import hashlib
import logging
import os
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from vitcore_audio.audio_preprocessing import load_dual_views
from vitcore_audio.model import ViTCoreAudio

CHECKPOINT_PATH = os.getenv("MODEL_WEIGHTS_PATH", "weights/vitcore_audio_best.pth")
CHECKPOINT_SHA256 = os.getenv("MODEL_WEIGHTS_SHA256", "").strip().lower()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

# analyze_audio is invoked from worker threads (see main.py's use of
# asyncio.to_thread) so concurrent requests don't block the event loop.
# The forward pass itself is stateless per-call (unlike FORENSICS, there's
# no shared attention cache to protect), but inference is still serialized
# here to bound peak memory under concurrent load on a single-GPU/CPU
# deployment — matching FORENSICS' own conservative default.
_inference_lock = threading.Lock()

# PyTorch 2.6 defaults torch.load to weights_only=True (a restricted
# unpickler that only allows a small set of known-safe types), specifically
# to stop a malicious checkpoint from executing arbitrary code on load. If
# this checkpoint's saved dict contains a raw numpy scalar (e.g. a metric
# like best_eer saved straight from sklearn without casting to a native
# Python float — see train.py's checkpoint dicts), that isn't on the
# default allowlist. Allowlisting exactly this one type — rather than
# disabling the safe loader entirely — keeps weights_only=True's protection
# intact for everything else in the file; the genuine "this checkpoint
# needs full unsafe unpickling" path below still stays gated behind
# ALLOW_UNTRUSTED_CHECKPOINT.
import numpy._core.multiarray

torch.serialization.add_safe_globals([np._core.multiarray.scalar])  # type: ignore[attr-defined]

# Matches datasets.py's eval-time (train=False) transform exactly: plain
# ToTensor (float32, [0,1], CHW), no Normalize, no augmentation. Any
# mismatch here would silently skew every prediction against what the
# checkpoint was actually trained/evaluated on.
_TO_TENSOR = transforms.ToTensor()


def _verify_checkpoint_integrity(path: str) -> None:
    """Hash the checkpoint and log it, so an operator can capture it once
    from a known-good load and pin it via MODEL_WEIGHTS_SHA256. If that env
    var is set, refuse to start on a mismatch — this is a forensics tool,
    so a corrupted download or a silently swapped/tampered checkpoint should
    fail loudly, not produce quietly-wrong verdicts."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    print(f"[ViT-CORE-Audio] Checkpoint SHA-256: {digest}")

    if CHECKPOINT_SHA256 and digest != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint integrity check failed for {path}: expected "
            f"{CHECKPOINT_SHA256}, got {digest}. Refusing to load a "
            f"checkpoint that doesn't match MODEL_WEIGHTS_SHA256."
        )


def load_model():
    global _model
    model = ViTCoreAudio(num_classes=2, pretrained=False)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ViT-CORE-Audio] Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using untrained weights.")
    else:
        _verify_checkpoint_integrity(CHECKPOINT_PATH)
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
            sd = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
        except Exception:
            # weights_only=True (safe, restricted unpickler) failed — the
            # checkpoint likely predates it or uses a non-tensor container.
            # Falling back to full pickle deserialization is a real code-
            # execution risk if the file isn't trusted, so require an
            # explicit opt-in rather than silently downgrading. The SHA-256
            # check above already guards against a corrupted/swapped file;
            # this guards against ever reaching unsafe deserialization
            # without the operator having deliberately allowed it.
            if os.getenv("ALLOW_UNTRUSTED_CHECKPOINT", "").strip().lower() not in ("1", "true", "yes"):
                raise RuntimeError(
                    f"Checkpoint at {CHECKPOINT_PATH} failed to load with weights_only=True "
                    "(the safe loader) and ALLOW_UNTRUSTED_CHECKPOINT is not set. Refusing to "
                    "fall back to full pickle deserialization of an unverified checkpoint. If "
                    "you trust this file's origin, set ALLOW_UNTRUSTED_CHECKPOINT=true."
                )
            logging.getLogger(__name__).warning(
                "weights_only=True failed — falling back to weights_only=False "
                "(ALLOW_UNTRUSTED_CHECKPOINT is set). Only safe with trusted checkpoints."
            )
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            sd = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
        model.load_state_dict(sd)

    model.to(DEVICE)
    model.eval()

    _model = model
    print(f"[ViT-CORE-Audio] Model loaded on {DEVICE}")


def get_model():
    if _model is None:
        load_model()
    return _model


def get_status() -> dict:
    """Liveness/readiness details for /health — whether the model is
    actually usable, not just whether the process is up."""
    return {
        "model_loaded": _model is not None,
        "device": str(DEVICE),
        "weights_path": CHECKPOINT_PATH,
        "weights_found": os.path.exists(CHECKPOINT_PATH),
    }


def _view_to_b64_jpeg(view: np.ndarray) -> str:
    """Render one 224x224x3 uint8 spectral view as a colorized JPEG for
    the frontend's Mel View / CQT View tabs. The 3 channels are identical
    replicas of the same grayscale magnitude (see vitcore_audio/audio_preprocessing.py's
    _to_uint8_image), so a single channel is colorized with a magma
    colormap and flipped vertically — spectrograms conventionally read
    low-frequency-at-bottom, but the stored array is row-0-at-top. This is
    purely a display transform; the model itself always sees the raw
    grayscale-replicated array via _TO_TENSOR, never this colorized copy."""
    gray = view[:, :, 0]
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
    colored = cv2.flip(colored, 0)
    _, buf = cv2.imencode(".jpg", colored)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@torch.inference_mode()
def analyze_audio(path: str, generate_visuals: bool = True) -> dict:
    """
    Runs the full dual-view pipeline on an audio file at `path` and
    returns a verdict. `path` should already be a file on disk (main.py
    writes the upload to a tempfile first, the same way FORENSICS'
    extract_frames_to_pil does for video).
    """
    model = get_model()
    start = time.time()

    mel_view, cqt_view = load_dual_views(path)

    mel_tensor = _TO_TENSOR(mel_view).unsqueeze(0).to(DEVICE)
    cqt_tensor = _TO_TENSOR(cqt_view).unsqueeze(0).to(DEVICE)

    with _inference_lock:
        logits, f1_norm, f2_norm = model(mel_tensor, cqt_tensor)

        # Cosine similarity between the two L2-normalized view embeddings —
        # both are already unit-norm, so the dot product IS the cosine
        # similarity. This is the audio analogue of FORENSICS' frame/face
        # quality signal: how much the mel and CQT renderings of this same
        # clip agree, in the representation space the model actually
        # reasons in. Low agreement on an otherwise-confident verdict is
        # worth flagging for manual review the same way a blurry face is.
        view_agreement = float(F.cosine_similarity(f1_norm, f2_norm).item())

        # LABEL_MAP in the training repo's datasets.py: 0 = bonafide, 1 = spoof
        probs = F.softmax(logits, dim=1)[0]
        spoof_prob = probs[1].item()

    is_spoof = spoof_prob >= 0.5

    visuals = {"mel": "", "cqt": ""}
    if generate_visuals:
        visuals["mel"] = _view_to_b64_jpeg(mel_view)
        visuals["cqt"] = _view_to_b64_jpeg(cqt_view)

    return {
        "probability": round(spoof_prob, 4),
        "verdict": "SPOOF" if is_spoof else "BONAFIDE",
        "confidence": round((spoof_prob if is_spoof else 1 - spoof_prob) * 100, 1),
        "view_agreement": round(view_agreement, 4),
        "is_low_confidence": 0.4 < spoof_prob < 0.6,
        "processing_time_sec": round(time.time() - start, 2),
        "visuals": visuals,
    }
