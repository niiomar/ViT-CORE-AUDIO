import numpy as np

from datasets import LABEL_MAP, AudioSpoofDataset
from tests.conftest import write_sine_wav


def _make_protocol_and_audio(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    write_sine_wav(str(audio_dir / "sample1.wav"), freq=220.0)
    write_sine_wav(str(audio_dir / "sample2.wav"), freq=440.0)

    protocol_path = tmp_path / "protocol.txt"
    protocol_path.write_text("SPK1 sample1 - SYS1 bonafide\nSPK2 sample2 - SYS2 spoof\n")
    return str(protocol_path), str(audio_dir)


def test_dataset_parses_protocol_and_returns_expected_item_shape(tmp_path):
    protocol_path, audio_dir = _make_protocol_and_audio(tmp_path)

    ds = AudioSpoofDataset(protocol_path, audio_dir, train=True, file_ext=".wav")
    assert len(ds) == 2

    item = ds[0]
    assert item["view1"].shape == (3, 224, 224)
    assert item["view2"].shape == (3, 224, 224)
    assert item["label"].item() == LABEL_MAP["bonafide"]
    assert item["filename"] == "sample1"

    item2 = ds[1]
    assert item2["label"].item() == LABEL_MAP["spoof"]


def test_dataset_eval_mode_skips_augmentation(tmp_path):
    protocol_path, audio_dir = _make_protocol_and_audio(tmp_path)
    ds = AudioSpoofDataset(protocol_path, audio_dir, train=False, file_ext=".wav")
    item = ds[0]
    assert item["view1"].shape == (3, 224, 224)


def test_cache_dir_populates_and_is_reused(tmp_path):
    protocol_path, audio_dir = _make_protocol_and_audio(tmp_path)
    cache_dir = tmp_path / "cache"

    ds = AudioSpoofDataset(protocol_path, audio_dir, train=False, file_ext=".wav", cache_dir=str(cache_dir))
    item_first = ds[0]

    cached_files = list(cache_dir.glob("*.npz"))
    assert len(cached_files) == 1

    # Second access must come from the cache and match the first
    # (train=False so there's no augmentation randomness to confound this).
    item_second = ds[0]
    assert np.array_equal(item_first["view1"].numpy(), item_second["view1"].numpy())
    assert np.array_equal(item_first["view2"].numpy(), item_second["view2"].numpy())
