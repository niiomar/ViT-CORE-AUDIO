import numpy as np

from audio_preprocessing import (
    DURATION_SECONDS,
    SAMPLE_RATE,
    _load_and_fix_length,
    load_dual_views,
    waveform_to_cqt_view,
    waveform_to_mel_view,
)
from tests.conftest import write_sine_wav


def test_dual_views_shape_dtype_and_range(synthetic_wav_path):
    mel_view, cqt_view = load_dual_views(synthetic_wav_path)
    for view in (mel_view, cqt_view):
        assert view.shape == (224, 224, 3)
        assert view.dtype == np.uint8
        assert view.min() >= 0 and view.max() <= 255


def test_cqt_stays_within_nyquist(synthetic_wav_path):
    """Regression test for the bug noted in the README: the original CQT
    config (224 bins @ 24 bins/octave) exceeded the Nyquist frequency at
    16kHz and raised librosa.ParameterError. The current config (84 bins
    @ 12 bins/octave) must not raise."""
    wav = _load_and_fix_length(synthetic_wav_path)
    cqt_view = waveform_to_cqt_view(wav, sr=SAMPLE_RATE)
    assert cqt_view.shape == (224, 224, 3)


def test_mel_view_runs_without_error(synthetic_wav_path):
    wav = _load_and_fix_length(synthetic_wav_path)
    mel_view = waveform_to_mel_view(wav, sr=SAMPLE_RATE)
    assert mel_view.shape == (224, 224, 3)


def test_fixed_length_short_clip_is_looped_to_target_duration(tmp_path):
    short_path = tmp_path / "short.wav"
    write_sine_wav(str(short_path), duration=0.05)  # much shorter than DURATION_SECONDS

    wav = _load_and_fix_length(str(short_path))
    assert len(wav) == int(SAMPLE_RATE * DURATION_SECONDS)


def test_fixed_length_long_clip_is_center_cropped(tmp_path):
    long_path = tmp_path / "long.wav"
    write_sine_wav(str(long_path), duration=DURATION_SECONDS * 3)

    wav = _load_and_fix_length(str(long_path))
    assert len(wav) == int(SAMPLE_RATE * DURATION_SECONDS)


def test_two_views_are_not_identical(synthetic_wav_path):
    """Mel and CQT are genuinely different transforms, not the same
    spectrogram twice — the consistency loss only means anything if this
    holds."""
    mel_view, cqt_view = load_dual_views(synthetic_wav_path)
    assert not np.array_equal(mel_view, cqt_view)
