"""
Dataset loader for audio deepfake detection, following the ASVspoof
protocol-file convention (the de facto standard format for this task —
ASVspoof 2019/2021 LA and the In-the-Wild dataset all use variants of
this speaker/file/system/label layout).

Expected protocol file format (whitespace-separated, no header):
    SPEAKER_ID  FILENAME  -  SYSTEM_ID  LABEL

where LABEL is "bonafide" or "spoof". Only FILENAME and LABEL are used
here; the other columns are read but ignored, matching how most public
baselines parse these files.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .audio_preprocessing import load_dual_views
from .augmentations import DFDC_Selim, RaAug

logger = logging.getLogger(__name__)

LABEL_MAP = {"bonafide": 0, "spoof": 1}  # 0 = real, 1 = fake — matches ViT-CORE's convention
MAX_LOAD_RETRIES = 5


class AudioSpoofDataset(Dataset):
    def __init__(
        self,
        protocol_path: str,
        audio_dir: str,
        *,
        train: bool = True,
        file_ext: str = ".flac",
        cache_dir: str | None = None,
    ):
        """
        protocol_path: path to the ASVspoof-style protocol .txt file
        audio_dir:     directory containing the audio files referenced by
                        the protocol's FILENAME column
        train:         if True, augmentations (RaAug/DFDC_Selim) are applied;
                        if False, only the raw dual-view transform runs —
                        matches ViT-CORE's train/eval augmentation split
        file_ext:      extension to append to FILENAME if the protocol
                        entries don't already include one (ASVspoof
                        protocols conventionally omit it)
        cache_dir:     if set, the (pre-augmentation) mel/CQT views are
                        cached to disk here on first computation and read
                        back on every subsequent access — the CQT
                        transform in particular is expensive enough that
                        recomputing it every epoch is the dominant cost of
                        training on a real dataset. Augmentations are
                        still applied fresh (randomized) after the cached
                        views are loaded, so caching doesn't affect
                        augmentation diversity. None disables caching.
        """
        self.audio_dir = audio_dir
        self.train = train
        self.file_ext = file_ext
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.entries: list[tuple[str, int]] = []

        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                filename, label_str = parts[1], parts[-1]
                if label_str not in LABEL_MAP:
                    continue
                self.entries.append((filename, LABEL_MAP[label_str]))

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_path(self, filename: str) -> str:
        if os.path.splitext(filename)[1]:
            return os.path.join(self.audio_dir, filename)
        return os.path.join(self.audio_dir, filename + self.file_ext)

    def _cache_path(self, filename: str) -> str:
        assert self.cache_dir is not None
        safe_name = filename.replace(os.sep, "_").replace("/", "_")
        return os.path.join(self.cache_dir, safe_name + ".npz")

    def _load_views(self, path: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.cache_dir:
            return load_dual_views(path)

        cache_path = self._cache_path(filename)
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return cached["mel"], cached["cqt"]

        mel_view, cqt_view = load_dual_views(path)

        # Write atomically (write-then-rename) so concurrent DataLoader
        # workers racing to populate the cache on the first epoch never
        # read a partially-written file.
        tmp_path = f"{cache_path}.{os.getpid()}.tmp.npz"
        np.savez(tmp_path, mel=mel_view, cqt=cqt_view)
        os.replace(tmp_path, cache_path)

        return mel_view, cqt_view

    def _safe_load_views(self, idx: int) -> tuple[np.ndarray, np.ndarray, str, int]:
        """Load dual views for entry `idx`, retrying at subsequent entries if the
        audio file is missing/corrupt/unreadable — a single bad file in a
        multi-thousand-file real dataset shouldn't kill an entire training run.
        Returns the (filename, label) actually loaded, which is NOT necessarily
        entries[idx] once a retry has happened — the caller must use these
        rather than re-reading entries[idx], or it'd mislabel the fallback
        sample with the broken entry's label."""
        next_idx = idx
        for _ in range(MAX_LOAD_RETRIES):
            filename, label = self.entries[next_idx]
            path = self._resolve_path(filename)
            try:
                mel_view, cqt_view = self._load_views(path, filename)
                return mel_view, cqt_view, filename, label
            except Exception as exc:
                logger.warning(f"Skipping unreadable audio file: {path} ({exc})")
                next_idx = (next_idx + 1) % len(self.entries)
        raise OSError(
            f"Could not find a readable audio file after {MAX_LOAD_RETRIES} attempts starting from index {idx}"
        )

    def __getitem__(self, idx: int):
        mel_view, cqt_view, filename, label = self._safe_load_views(idx)

        if self.train:
            mel_view = RaAug(mel_view)
            cqt_view = DFDC_Selim(cqt_view)

        mel_tensor = self.to_tensor(mel_view)  # -> float32, [0,1], CHW
        cqt_tensor = self.to_tensor(cqt_view)

        return {
            "view1": mel_tensor,
            "view2": cqt_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "filename": filename,
        }
