"""
Reproducibility and fail-fast helpers shared by train.py and evaluate.py.
Ported from ViT-CORE's train.py so ViT-CORE-Audio's training pipeline
matches it: seeded runs, and upfront validation of every input path so a
typo surfaces immediately instead of after the first batch (or, worse,
after the first several minutes of spectrogram caching).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_paths(paths: dict[str, str]) -> None:
    """Raise a clear error up front listing every path in `paths` (name -> path) that doesn't exist."""
    missing = [f"  --{name}: {path}" for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("The following paths do not exist:\n" + "\n".join(missing))
