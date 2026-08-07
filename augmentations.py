"""
Dual-view augmentations for audio, mirroring the naming convention from
ViT-CORE's augmentations.py — RaAug and DFDC_Selim were two independently
randomized augmentation pipelines applied to the two visual crops.
Here, RaAug augments the mel view and DFDC_Selim augments the CQT view;
each is applied AFTER audio_preprocessing.py's transform, on the
resulting spectrogram image, not on the raw waveform — this keeps the
two views' augmentation independent even though the underlying
transform (mel vs CQT) already gives them different starting points.
"""
from __future__ import annotations

import random

import numpy as np


def _spec_augment(spec: np.ndarray, *, freq_mask_pct: float, time_mask_pct: float, n_masks: int) -> np.ndarray:
    """
    SpecAugment-style masking (Park et al., 2019) — zero out random
    frequency bands and time windows. This is the audio-spectrogram
    analogue of visual random erasing/cutout, and is the single most
    standard augmentation in speech model training.
    """
    spec = spec.copy()
    h, w = spec.shape[:2]

    for _ in range(n_masks):
        f_width = int(h * freq_mask_pct * random.random())
        if f_width > 0:
            f0 = random.randint(0, max(1, h - f_width))
            spec[f0:f0 + f_width, :, :] = 0

        t_width = int(w * time_mask_pct * random.random())
        if t_width > 0:
            t0 = random.randint(0, max(1, w - t_width))
            spec[:, t0:t0 + t_width, :] = 0

    return spec


def _additive_gaussian_noise(spec: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0, std, spec.shape).astype(np.float32)
    out = spec.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _brightness_jitter(spec: np.ndarray, max_delta: float) -> np.ndarray:
    """Analogue of the visual pipeline's illumination jitter — scales
    overall spectrogram intensity, simulating recording-level/gain
    variation between source devices."""
    factor = 1.0 + random.uniform(-max_delta, max_delta)
    out = spec.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def RaAug(mel_view: np.ndarray) -> np.ndarray:
    """
    Augmentation pipeline for View 1 (mel spectrogram). Randomized,
    moderate-strength — mirrors ViT-CORE's RaAug as the "primary" view
    augmentation.
    """
    out = mel_view
    if random.random() < 0.8:
        out = _spec_augment(out, freq_mask_pct=0.15, time_mask_pct=0.15, n_masks=2)
    if random.random() < 0.5:
        out = _additive_gaussian_noise(out, std=5.0)
    if random.random() < 0.5:
        out = _brightness_jitter(out, max_delta=0.15)
    return out


def DFDC_Selim(cqt_view: np.ndarray) -> np.ndarray:
    """
    Augmentation pipeline for View 2 (CQT). Deliberately a DIFFERENT
    randomization (different probabilities/strengths) from RaAug — per
    ViT-CORE's design note, using the same augmentation on both views
    would let the model learn a trivial shortcut (match identical noise
    patterns rather than semantic content), defeating the point of the
    consistency loss.
    """
    out = cqt_view
    if random.random() < 0.6:
        out = _spec_augment(out, freq_mask_pct=0.20, time_mask_pct=0.10, n_masks=1)
    if random.random() < 0.7:
        out = _additive_gaussian_noise(out, std=8.0)
    if random.random() < 0.3:
        out = _brightness_jitter(out, max_delta=0.25)
    return out
