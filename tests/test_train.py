import torch

from train import (
    build_checkpoint,
    build_scheduler,
    compute_class_weights,
    load_checkpoint,
    should_stop_early,
)
from vitcore_audio.model import ModelEma, ViTCoreAudio


def test_should_stop_early():
    assert should_stop_early(epochs_since_improvement=10, patience=10) is True
    assert should_stop_early(epochs_since_improvement=9, patience=10) is False
    assert should_stop_early(epochs_since_improvement=100, patience=0) is False  # 0 disables


def test_build_scheduler_warmup_then_decays_to_floor():
    model = torch.nn.Linear(2, 2)
    lr, min_lr, epochs, warmup_epochs = 1e-3, 1e-6, 10, 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs, min_lr)

    lrs = []
    for _ in range(epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert lrs[0] < lr  # warmup starts below the base LR
    assert lrs[warmup_epochs] == lr  # warmup reaches the base LR by its end
    assert lrs[-1] < lrs[warmup_epochs]  # cosine decay brings it back down


def test_build_scheduler_with_zero_warmup_epochs_still_works():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, epochs=5, warmup_epochs=0, min_lr=1e-6)
    for _ in range(5):
        scheduler.step()  # must not raise


class _StubEntries:
    def __init__(self, labels):
        self.entries = [(f"f{i}", label) for i, label in enumerate(labels)]


def test_compute_class_weights_upweights_the_minority_class():
    dataset = _StubEntries([0, 1, 1, 1])  # 1 bonafide, 3 spoof
    # _StubEntries duck-types AudioSpoofDataset (compute_class_weights only reads .entries).
    weights = compute_class_weights(dataset, device=torch.device("cpu"))
    assert weights[0] > weights[1]  # bonafide (minority) gets the larger weight


def test_checkpoint_round_trip_restores_training_state(tmp_path):
    model = ViTCoreAudio(num_classes=2, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, epochs=10, warmup_epochs=2, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    ema = ModelEma(model, decay=0.9)

    ckpt = build_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=3,
        val_eer=0.12,
        best_eer=0.10,
        val_metrics={"eer": 0.12},
        ema=ema,
    )
    assert "ema" in ckpt

    fresh_model = ViTCoreAudio(num_classes=2, pretrained=False)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_scheduler = build_scheduler(fresh_optimizer, epochs=10, warmup_epochs=2, min_lr=1e-6)
    fresh_scaler = torch.amp.GradScaler("cpu", enabled=False)

    ckpt_path = tmp_path / "ckpt_roundtrip.pth"
    torch.save(ckpt, ckpt_path)
    loaded_epoch, best_eer, ema_state = load_checkpoint(
        str(ckpt_path),
        fresh_model,
        fresh_optimizer,
        fresh_scheduler,
        fresh_scaler,
        device=torch.device("cpu"),
    )

    # load_checkpoint returns the raw stored epoch — whether the caller should
    # resume AT it (mid-epoch) or AFTER it (+1, epoch fully completed) depends
    # on last_batch.txt, which main() checks, not load_checkpoint itself.
    assert loaded_epoch == 3
    assert best_eer == 0.10
    assert ema_state is not None
    for p_a, p_b in zip(model.parameters(), fresh_model.parameters(), strict=True):
        assert torch.equal(p_a, p_b)
