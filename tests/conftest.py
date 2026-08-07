import numpy as np
import pytest
import soundfile as sf

SAMPLE_RATE = 16000


def write_sine_wav(path: str, *, sr: int = SAMPLE_RATE, duration: float = 2.0, freq: float = 220.0) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, wav, sr)


@pytest.fixture
def synthetic_wav_path(tmp_path) -> str:
    """A short real (decodable) sine-wave clip — enough to exercise the
    full librosa mel/CQT pipeline without needing a real dataset."""
    path = tmp_path / "synthetic.wav"
    write_sine_wav(str(path))
    return str(path)
