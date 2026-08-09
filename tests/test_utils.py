import random

import numpy as np
import pytest
import torch

from utils import seed_worker, set_seed, validate_paths


def test_set_seed_makes_random_sequences_reproducible():
    set_seed(123)
    a = (torch.rand(4), np.random.rand(4), random.random())

    set_seed(123)
    b = (torch.rand(4), np.random.rand(4), random.random())

    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_seed_worker_is_deterministic_for_a_given_initial_seed():
    torch.manual_seed(42)
    seed_worker(0)
    state_a = (np.random.rand(4), random.random())

    torch.manual_seed(42)
    seed_worker(0)
    state_b = (np.random.rand(4), random.random())

    assert np.array_equal(state_a[0], state_b[0])
    assert state_a[1] == state_b[1]


def test_validate_paths_passes_when_all_paths_exist(tmp_path):
    f = tmp_path / "exists.txt"
    f.write_text("x")
    validate_paths({"some_file": str(f), "some_dir": str(tmp_path)})


def test_validate_paths_raises_listing_every_missing_path(tmp_path):
    present = tmp_path / "present.txt"
    present.write_text("x")
    missing = str(tmp_path / "missing.txt")

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_paths({"present": str(present), "missing": missing})

    assert "missing" in str(exc_info.value)
    assert missing in str(exc_info.value)
    assert "present" not in str(exc_info.value)
