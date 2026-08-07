"""
Training loop for ViT-CORE-Audio. Structurally identical to ViT-CORE's
train.py — same optimizer/scheduler pattern, same checkpoint-on-best-EER
logic (adapted from checkpoint-on-best-val-accuracy, since EER is this
domain's actual model-selection metric).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import AudioSpoofDataset
from loss import ViTCoreAudioLoss
from metrics import compute_all
from model import ViTCoreAudio


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    running = {"total": 0.0, "classification": 0.0, "consistency": 0.0}

    for batch in loader:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, f1_norm, f2_norm = model(view1, view2)
            losses = loss_fn(logits, f1_norm, f2_norm, labels)
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        for k in running:
            running[k] += losses[k].item() * labels.size(0)

    n = len(loader.dataset)  # type: ignore[arg-type]  # torch's Dataset stub doesn't declare __len__, ours has one
    return {k: v / n for k, v in running.items()}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    all_labels: list[int] = []
    all_scores: list[float] = []
    all_preds: list[int] = []

    for batch in loader:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        labels = batch["label"]

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, _, _ = model(view1, view2)
        probs = F.softmax(logits.float(), dim=1)[:, 1]  # P(spoof)
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.numpy())
        all_scores.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return compute_all(all_labels, all_scores, all_preds)


def build_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup (avoids destabilizing the pretrained ViT backbone in
    the first few steps) followed by cosine annealing for the remainder."""
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def compute_class_weights(dataset: AudioSpoofDataset, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights (the standard 'balanced' formula:
    N / (num_classes * count_i)) — ASVspoof-style protocols are typically
    dominated by spoof samples, so unweighted CE biases toward predicting
    the majority class."""
    labels = np.array([label for _, label in dataset.entries])
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = counts.sum() / (len(counts) * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--train_audio_dir", required=True)
    parser.add_argument("--val_protocol", required=True)
    parser.add_argument("--val_audio_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--warmup_epochs", type=int, default=2, help="linear LR warmup epochs before cosine annealing begins"
    )
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="directory to cache precomputed mel/CQT views (train and val use separate "
        "subdirectories); omit to disable caching and recompute every epoch",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="initialize the ViT-S/16 backbone from ImageNet-pretrained weights (default)",
    )
    parser.add_argument(
        "--no_pretrained",
        dest="pretrained",
        action="store_false",
        help="train the backbone from random initialization instead",
    )
    parser.add_argument(
        "--no_amp", action="store_true", help="disable automatic mixed precision (AMP is on by default on CUDA)"
    )
    parser.add_argument(
        "--class_weighted_loss",
        action="store_true",
        help="weight the classification loss inversely to class frequency in the "
        "training protocol (recommended for ASVspoof-style imbalanced splits)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="path to a checkpoint to resume training from (model/optimizer/scheduler/scaler state)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_cache_dir = os.path.join(args.cache_dir, "train") if args.cache_dir else None
    val_cache_dir = os.path.join(args.cache_dir, "val") if args.cache_dir else None
    train_ds = AudioSpoofDataset(args.train_protocol, args.train_audio_dir, train=True, cache_dir=train_cache_dir)
    val_ds = AudioSpoofDataset(args.val_protocol, args.val_audio_dir, train=False, cache_dir=val_cache_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = ViTCoreAudio(num_classes=2, pretrained=args.pretrained).to(device)

    class_weights = None
    if args.class_weighted_loss:
        class_weights = compute_class_weights(train_ds, device)
        print(f"Class-weighted loss enabled: bonafide={class_weights[0]:.3f}, spoof={class_weights[1]:.3f}")

    loss_fn = ViTCoreAudioLoss(consistency_weight=args.consistency_weight, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_eer = float("inf")

    if args.resume:
        # weights_only=True: this checkpoint only holds tensors/numbers
        # (state dicts + scalars), so there's no need to unpickle
        # arbitrary objects to load it.
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_eer = ckpt["best_eer"]
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best val EER so far {best_eer * 100:.2f}%)")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler, use_amp)
        val_metrics = evaluate(model, val_loader, device, use_amp)
        scheduler.step()

        is_best = val_metrics["eer"] < best_eer
        if is_best:
            best_eer = val_metrics["eer"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['total']:.4f} "
            f"(cls={train_metrics['classification']:.4f}, cons={train_metrics['consistency']:.4f}) | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_eer={val_metrics['eer_pct']:.2f}%"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "val_eer": val_metrics["eer"],
            "best_eer": best_eer,
            "val_metrics": val_metrics,
        }

        # Saved every epoch so a crash mid-run can resume with --resume,
        # not just re-scored from the best checkpoint.
        last_ckpt_path = os.path.join(args.checkpoint_dir, "vitcore_audio_last.pth")
        torch.save(ckpt, last_ckpt_path)

        if is_best:
            best_ckpt_path = os.path.join(args.checkpoint_dir, "vitcore_audio_best.pth")
            torch.save(ckpt, best_ckpt_path)
            print(f"  -> new best EER {best_eer * 100:.2f}%, saved to {best_ckpt_path}")

    print(f"Training complete. Best val EER: {best_eer * 100:.2f}%")


if __name__ == "__main__":
    main()
