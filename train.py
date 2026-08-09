"""
Training loop for ViT-CORE-Audio, replicating ViT-CORE's train.py pipeline
end-to-end: seeded runs, class-balanced sampling, decoupled weight decay,
label smoothing, gradient clipping, mixed precision, warmup+cosine LR with
a floor, EMA of weights, early stopping, TensorBoard + CSV logging, and
crash-safe checkpointing (latest/best/exit, all resumable).

Differs from ViT-CORE only where the domain requires it: EER (not AUC) is
the model-selection/early-stopping metric, per this project's own
argument for EER as the field-standard metric (see README); and training
auto-resumes from <checkpoint_dir>/vitcore_audio_latest.pth when present,
rather than requiring a fixed --output-dir layout.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import logging
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from datasets import AudioSpoofDataset
from loss import ViTCoreAudioLoss
from metrics import compute_all
from model import ModelEma, ViTCoreAudio, build_param_groups
from utils import seed_worker, set_seed, validate_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_protocol", required=True)
    p.add_argument("--train_audio_dir", required=True)
    p.add_argument("--val_protocol", required=True)
    p.add_argument("--val_audio_dir", required=True)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=1e-6, help="LR floor at the end of cosine decay")
    p.add_argument(
        "--warmup_epochs", type=int, default=2, help="linear LR warmup epochs before cosine annealing begins"
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="AdamW weight decay (skipped for biases and 1-D norm params)",
    )
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--consistency_weight", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--cache_dir",
        default=None,
        help="directory to cache precomputed mel/CQT views (train and val use separate "
        "subdirectories); omit to disable caching and recompute every epoch",
    )
    p.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="initialize the ViT-S/16 backbone from ImageNet-pretrained weights (default)",
    )
    p.add_argument(
        "--no_pretrained",
        dest="pretrained",
        action="store_false",
        help="train the backbone from random initialization instead",
    )
    p.add_argument(
        "--no_amp", action="store_true", help="disable automatic mixed precision (AMP is on by default on CUDA)"
    )
    p.add_argument(
        "--balanced_sampling",
        dest="balanced_sampling",
        action="store_true",
        default=True,
        help="sample training batches with class-balanced probability via WeightedRandomSampler (default: on)",
    )
    p.add_argument(
        "--no_balanced_sampling",
        dest="balanced_sampling",
        action="store_false",
        help="disable balanced sampling — use with --class_weighted_loss instead, not both",
    )
    p.add_argument(
        "--class_weighted_loss",
        action="store_true",
        help="weight the classification loss inversely to class frequency in the training protocol "
        "(off by default — redundant with the default --balanced_sampling; use one or the other, not both)",
    )
    p.add_argument(
        "--ema",
        dest="ema",
        action="store_true",
        default=True,
        help="track an EMA of weights for validation/checkpointing (default: on)",
    )
    p.add_argument("--no_ema", dest="ema", action="store_false")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="stop after this many epochs with no val-EER improvement (0 disables)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        default=None,
        help="checkpoint path to resume from; if omitted, auto-resumes from "
        "<checkpoint_dir>/vitcore_audio_latest.pth when it exists",
    )
    return p.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_clip_norm: float,
    ema: ModelEma | None,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train()
    running = {"total": 0.0, "classification": 0.0, "consistency": 0.0}

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}")
    for batch in pbar:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, f1_norm, f2_norm = model(view1, view2)
            losses = loss_fn(logits, f1_norm, f2_norm, labels)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        for k in running:
            running[k] += losses[k].item() * labels.size(0)
        pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

    n = len(loader.dataset)  # type: ignore[arg-type]  # torch's Dataset stub doesn't declare __len__, ours has one
    return {k: v / n for k, v in running.items()}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    all_labels: list[int] = []
    all_scores: list[float] = []
    all_preds: list[int] = []

    for batch in tqdm(loader, desc="Validating", leave=False):
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
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, min_lr: float
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup (avoids destabilizing the pretrained ViT backbone in the first few
    steps) followed by cosine annealing down to a floor of `min_lr` for the remainder."""
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
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


def should_stop_early(epochs_since_improvement: int, patience: int) -> bool:
    """True once `patience` epochs have passed with no val-EER improvement. patience <= 0 disables."""
    return patience > 0 and epochs_since_improvement >= patience


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    val_eer: float,
    best_eer: float,
    val_metrics: dict,
    ema: ModelEma | None,
) -> dict:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "val_eer": val_eer,
        "best_eer": best_eer,
        "val_metrics": val_metrics,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    return ckpt


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, dict | None]:
    # weights_only=True: this checkpoint only holds tensors/numbers (state
    # dicts + scalars), so there's no need to unpickle arbitrary objects.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"] + 1, ckpt["best_eer"], ckpt.get("ema")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_paths(
        {
            "train_protocol": args.train_protocol,
            "train_audio_dir": args.train_audio_dir,
            "val_protocol": args.val_protocol,
            "val_audio_dir": args.val_audio_dir,
        }
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    logger.info(f"Device: {device}  AMP: {use_amp}  EMA: {args.ema}  Balanced sampling: {args.balanced_sampling}")

    train_cache_dir = os.path.join(args.cache_dir, "train") if args.cache_dir else None
    val_cache_dir = os.path.join(args.cache_dir, "val") if args.cache_dir else None
    train_ds = AudioSpoofDataset(args.train_protocol, args.train_audio_dir, train=True, cache_dir=train_cache_dir)
    val_ds = AudioSpoofDataset(args.val_protocol, args.val_audio_dir, train=False, cache_dir=val_cache_dir)

    loader_kwargs: dict = {
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": (args.num_workers > 0),
    }
    if args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker
        loader_kwargs["generator"] = torch.Generator().manual_seed(args.seed)

    train_sampler = None
    train_shuffle = True
    if args.balanced_sampling:
        labels = [label for _, label in train_ds.entries]
        counts = Counter(labels)
        sample_weights = [1.0 / counts[label] for label in labels]
        train_sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(args.seed)
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = ViTCoreAudio(num_classes=2, pretrained=args.pretrained).to(device)

    class_weights = None
    if args.class_weighted_loss:
        class_weights = compute_class_weights(train_ds, device)
        logger.info(f"Class-weighted loss enabled: bonafide={class_weights[0]:.3f}, spoof={class_weights[1]:.3f}")

    loss_fn = ViTCoreAudioLoss(
        consistency_weight=args.consistency_weight,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    latest_ckpt_path = os.path.join(args.checkpoint_dir, "vitcore_audio_latest.pth")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "vitcore_audio_best.pth")
    exit_ckpt_path = os.path.join(args.checkpoint_dir, "vitcore_audio_exit.pth")
    csv_path = os.path.join(args.checkpoint_dir, "vitcore_audio_losses.csv")

    resume_path = args.resume or (latest_ckpt_path if os.path.exists(latest_ckpt_path) else None)
    start_epoch, best_eer, ema_state = 1, float("inf"), None
    if resume_path:
        start_epoch, best_eer, ema_state = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)
        logger.info(f"Resumed from {resume_path} at epoch {start_epoch} (best val EER so far {best_eer * 100:.2f}%)")

    ema = None
    if args.ema:
        ema = ModelEma(model, decay=args.ema_decay).to(device)
        if ema_state is not None:
            ema.load_state_dict(ema_state)
    eval_model = ema.module if ema is not None else model

    writer = SummaryWriter(os.path.join(args.checkpoint_dir, "tensorboard"))
    atexit.register(writer.close)

    exit_state: dict = {"epoch": start_epoch - 1, "best_eer": best_eer, "val_eer": float("nan"), "val_metrics": {}}

    def save_on_exit() -> None:
        ckpt = build_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            exit_state["epoch"],
            exit_state["val_eer"],
            exit_state["best_eer"],
            exit_state["val_metrics"],
            ema,
        )
        torch.save(ckpt, exit_ckpt_path)
        logger.info(f"Saved exit checkpoint to {exit_ckpt_path}")

    atexit.register(save_on_exit)

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "total_loss", "cls_loss", "cons_loss", "val_accuracy", "val_auc", "val_eer_pct", "lr"]
            )

    epochs_since_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scaler,
            use_amp,
            args.grad_clip_norm,
            ema,
            epoch,
            args.epochs,
        )
        val_metrics = evaluate(eval_model, val_loader, device, use_amp)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_metrics["eer"] < best_eer
        if is_best:
            best_eer = val_metrics["eer"]
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['total']:.4f} "
            f"(cls={train_metrics['classification']:.4f}, cons={train_metrics['consistency']:.4f}) | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_eer={val_metrics['eer_pct']:.2f}% "
            f"lr={current_lr:.2e}"
        )

        writer.add_scalar("train/loss_total", train_metrics["total"], epoch)
        writer.add_scalar("train/loss_classification", train_metrics["classification"], epoch)
        writer.add_scalar("train/loss_consistency", train_metrics["consistency"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/auc", val_metrics["auc"], epoch)
        writer.add_scalar("val/eer_pct", val_metrics["eer_pct"], epoch)
        writer.add_scalar("lr", current_lr, epoch)

        ckpt = build_checkpoint(
            model, optimizer, scheduler, scaler, epoch, val_metrics["eer"], best_eer, val_metrics, ema
        )
        torch.save(ckpt, latest_ckpt_path)
        if is_best:
            torch.save(ckpt, best_ckpt_path)
            logger.info(f"  -> new best EER {best_eer * 100:.2f}%, saved to {best_ckpt_path}")

        exit_state.update(epoch=epoch, best_eer=best_eer, val_eer=val_metrics["eer"], val_metrics=val_metrics)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    round(train_metrics["total"], 4),
                    round(train_metrics["classification"], 4),
                    round(train_metrics["consistency"], 4),
                    round(val_metrics["accuracy"], 4),
                    round(val_metrics["auc"], 4),
                    round(val_metrics["eer_pct"], 2),
                    round(current_lr, 8),
                ]
            )

        if should_stop_early(epochs_since_improvement, args.early_stopping_patience):
            logger.info(f"No val-EER improvement in {epochs_since_improvement} epochs, stopping early.")
            break

    logger.info(f"Training complete. Best val EER: {best_eer * 100:.2f}%")


if __name__ == "__main__":
    main()
