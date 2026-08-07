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

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from audio_preprocessing import load_dual_views
from augmentations import DFDC_Selim, RaAug

LABEL_MAP = {"bonafide": 0, "spoof": 1}  # 0 = real, 1 = fake — matches ViT-CORE's convention


class AudioSpoofDataset(Dataset):
    def __init__(self, protocol_path: str, audio_dir: str, *, train: bool = True, file_ext: str = ".flac"):
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
        """
        self.audio_dir = audio_dir
        self.train = train
        self.file_ext = file_ext
        self.entries: list[tuple[str, int]] = []

        with open(protocol_path, "r") as f:
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

    def __getitem__(self, idx: int):
        filename, label = self.entries[idx]
        path = self._resolve_path(filename)

        mel_view, cqt_view = load_dual_views(path)

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
