"""
Smoke test: verifies the full dual-view inference pipeline runs end-to-end
on CPU with untrained weights (no checkpoint present in CI). This won't
catch accuracy regressions, but it catches import errors, shape mismatches,
and preprocessing/model contract drift before they hit main.
"""

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Force no checkpoint found (CI doesn't have weights) — matches FORENSICS'
# test_smoke.py convention of exercising the untrained-weights fallback path.
os.environ["MODEL_WEIGHTS_PATH"] = "nonexistent.pth"

import model as vitcore_audio_model


def _write_synthetic_wav(path: str, seconds: float = 4.0, sr: int = 16000, freq: float = 220.0) -> None:
    """A pure sine tone stands in for a real voice clip — this test only
    needs *some* fixed-length waveform that librosa can decode end-to-end,
    not anything realistic (there's no trained model here to be realistic
    against yet)."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, wav, sr)


def test_pipeline_runs_full_path():
    """Exercises audio_preprocessing's dual-view transform, the shared
    ViT-S/16 forward pass, and the mel/CQT view-agreement + visualization
    code directly — the class of regression this smoke test exists for."""
    vitcore_audio_model.load_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        _write_synthetic_wav(tmp.name)
        result = vitcore_audio_model.analyze_audio(tmp.name, generate_visuals=True)
    os.remove(tmp.name)

    assert 0.0 <= result["probability"] <= 1.0
    assert result["verdict"] in ("BONAFIDE", "SPOOF")
    assert 0.0 <= result["confidence"] <= 100.0
    # Both views' embeddings are L2-normalized, so cosine similarity is
    # bounded in [-1, 1] — anything outside that range means the
    # normalization or the similarity computation itself is broken.
    assert -1.0 <= result["view_agreement"] <= 1.0
    assert result["processing_time_sec"] >= 0.0

    assert isinstance(result["visuals"]["mel"], str) and result["visuals"]["mel"] != ""
    assert isinstance(result["visuals"]["cqt"], str) and result["visuals"]["cqt"] != ""


def test_visuals_omitted_when_not_requested():
    """Batch requests default explain=False — visuals should stay empty
    strings rather than being computed and discarded."""
    vitcore_audio_model.load_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        _write_synthetic_wav(tmp.name)
        result = vitcore_audio_model.analyze_audio(tmp.name, generate_visuals=False)
    os.remove(tmp.name)

    assert result["visuals"] == {"mel": "", "cqt": ""}


def test_status_reports_untrained_fallback():
    vitcore_audio_model.load_model()
    status = vitcore_audio_model.get_status()

    assert status["model_loaded"] is True
    assert status["weights_found"] is False
