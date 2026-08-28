import numpy as np

from vitcore_audio.augmentations import DFDC_Selim, RaAug


def _random_spec() -> np.ndarray:
    rng = np.random.RandomState(0)
    return (rng.rand(224, 224, 3) * 255).astype(np.uint8)


def test_raaug_preserves_shape_and_dtype():
    spec = _random_spec()
    out = RaAug(spec)
    assert out.shape == spec.shape
    assert out.dtype == spec.dtype


def test_dfdc_selim_preserves_shape_and_dtype():
    spec = _random_spec()
    out = DFDC_Selim(spec)
    assert out.shape == spec.shape
    assert out.dtype == spec.dtype


def test_raaug_actually_modifies_pixels():
    """Confirms the augmentation isn't a silent no-op. Individual calls
    are probabilistic (each sub-augmentation can roll to skip), so run
    enough trials that at least one produces a visible change."""
    spec = _random_spec()
    assert any(not np.array_equal(RaAug(spec), spec) for _ in range(30))


def test_dfdc_selim_actually_modifies_pixels():
    spec = _random_spec()
    assert any(not np.array_equal(DFDC_Selim(spec), spec) for _ in range(30))


def test_raaug_and_dfdc_selim_use_different_randomization():
    """RaAug and DFDC_Selim must not be the same augmentation pipeline —
    applying identical augmentation to both views would let the model
    shortcut on matching noise patterns instead of learning shared
    semantic content (per the design note in augmentations.py)."""
    import random

    spec = _random_spec()
    outputs_match_every_time = True
    for seed in range(30):
        random.seed(seed)
        out_ra = RaAug(spec)
        random.seed(seed)
        out_dfdc = DFDC_Selim(spec)
        if not np.array_equal(out_ra, out_dfdc):
            outputs_match_every_time = False
            break

    assert not outputs_match_every_time
