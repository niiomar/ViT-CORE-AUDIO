from tests.conftest import write_sine_wav
from vitcore_audio.datasets import LABEL_MAP, AudioSpoofDataset
from vitcore_audio.datasets_preprocessed import PreprocessedAudioSpoofDataset


def _make_protocol_and_populated_cache(tmp_path):
    """Mirrors the intended real workflow: AudioSpoofDataset(..., cache_dir=X)
    populates X, then PreprocessedAudioSpoofDataset reads back from X only."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    write_sine_wav(str(audio_dir / "sample1.wav"), freq=220.0)
    write_sine_wav(str(audio_dir / "sample2.wav"), freq=440.0)

    protocol_path = tmp_path / "protocol.txt"
    protocol_path.write_text("SPK1 sample1 - SYS1 bonafide\nSPK2 sample2 - SYS2 spoof\n")

    cache_dir = tmp_path / "cache"
    populate_ds = AudioSpoofDataset(
        str(protocol_path), str(audio_dir), train=False, file_ext=".wav", cache_dir=str(cache_dir)
    )
    for i in range(len(populate_ds)):
        populate_ds[i]  # triggers _load_views -> writes the .npz cache as a side effect

    return str(protocol_path), str(cache_dir)


def test_reads_back_the_same_views_a_populate_pass_cached(tmp_path):
    protocol_path, cache_dir = _make_protocol_and_populated_cache(tmp_path)

    ds = PreprocessedAudioSpoofDataset(protocol_path, cache_dir, train=False)
    assert len(ds) == 2

    item = ds[0]
    assert item["view1"].shape == (3, 224, 224)
    assert item["view2"].shape == (3, 224, 224)
    assert item["label"].item() == LABEL_MAP["bonafide"]
    assert item["filename"] == "sample1"


def test_train_mode_still_applies_augmentation_on_top_of_cached_views(tmp_path):
    protocol_path, cache_dir = _make_protocol_and_populated_cache(tmp_path)
    ds = PreprocessedAudioSpoofDataset(protocol_path, cache_dir, train=True)
    item = ds[0]
    assert item["view1"].shape == (3, 224, 224)


def test_missing_cache_entry_raises_rather_than_silently_computing(tmp_path):
    """The whole point of this class is to never decode raw audio — a
    missing cache entry is a data-pipeline error, not something to
    silently fall back and compute (that's what AudioSpoofDataset is for)."""
    protocol_path = tmp_path / "protocol.txt"
    protocol_path.write_text("SPK1 nonexistent - SYS1 bonafide\n")
    empty_cache_dir = tmp_path / "empty_cache"
    empty_cache_dir.mkdir()

    ds = PreprocessedAudioSpoofDataset(str(protocol_path), str(empty_cache_dir), train=False)
    # AudioSpoofDataset's retry wrapper (_safe_load_views) retries on any
    # exception up to MAX_LOAD_RETRIES times across the (single-entry)
    # dataset, then re-raises as OSError — this is inherited, unmodified,
    # tested behavior, not something this test needs to re-verify beyond
    # confirming it still surfaces as a clear failure rather than success.
    try:
        ds[0]
        raised = False
    except OSError:
        raised = True
    assert raised
