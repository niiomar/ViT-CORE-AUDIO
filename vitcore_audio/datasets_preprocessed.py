"""
PreprocessedAudioSpoofDataset: reads mel/CQT views exclusively from a
pre-populated cache directory of .npz files — never decodes raw audio or
computes spectrograms on the fly.

This file did not exist in the repo even though train.py imports it
unconditionally (`from vitcore_audio.datasets_preprocessed import
PreprocessedAudioSpoofDataset`) — a pre-existing gap, unrelated to and
predating the backend/frontend service work. Reconstructed here rather
than left broken, since train.py can't be imported at all without it.

The intended workflow (inferred from train.py's --preprocessed flag and
AudioSpoofDataset's existing cache_dir mechanism, since there was no
other spec to go on): run AudioSpoofDataset(..., cache_dir=X) once to
populate X with every entry's precomputed (mel, cqt) views as .npz files
(the exact format AudioSpoofDataset._load_views already reads/writes —
see its cache_dir docstring), then point PreprocessedAudioSpoofDataset at
that same directory for subsequent training runs. This is what makes
--preprocessed useful for large-scale training in the first place: the
CQT transform is the expensive part of audio_preprocessing.py, and this
class guarantees it's never recomputed, only ever read back.

`audio_dir` in the constructor therefore means "the precomputed cache
directory," not a directory of raw audio files — same overload train.py's
own --train_audio_dir/--val_audio_dir flags rely on when --preprocessed
is passed (there's no separate --preprocessed_dir flag).
"""

from __future__ import annotations

import os

import numpy as np

from .datasets import AudioSpoofDataset


class PreprocessedAudioSpoofDataset(AudioSpoofDataset):
    def __init__(self, protocol_path: str, cache_dir: str, *, train: bool = True):
        # cache_dir is passed as both audio_dir and cache_dir: _resolve_path
        # (which would build a raw-audio path) is never actually called by
        # our _load_views override below, so its output being meaningless
        # here is harmless dead computation, not a bug — only _cache_path
        # (which needs self.cache_dir set) is used.
        super().__init__(protocol_path, audio_dir=cache_dir, train=train, cache_dir=cache_dir)

    def _load_views(self, path: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
        cache_path = self._cache_path(filename)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No precomputed cache for {filename!r} at {cache_path}. "
                "PreprocessedAudioSpoofDataset never computes views on the fly — "
                "populate the cache first with AudioSpoofDataset(..., cache_dir=...)."
            )
        cached = np.load(cache_path)
        return cached["mel"], cached["cqt"]
