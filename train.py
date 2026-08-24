"""
Training loop for ViT-CORE-Audio.

Checkpoint/resume: model/optimizer/scheduler/running-loss state saved
every 100 batches to --checkpoint_dir. If interrupted, rerun the same
command — it resumes from the last saved batch, correctly, without
re-loading any data for batches already completed (see ResumableSampler).
"""
from __future__ import annotations

import argparse
import atexit
import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from datasets_preprocessed import PreprocessedAudioSpoofDataset
from loss import ViTCoreAudioLoss
from metrics import compute_all
from model import ViTCoreAudio

CHECKPOINT_EVERY_N_BATCHES = 100


class ResumableSampler(Sampler):
    """
    Deterministic per-epoch shuffle (seeded by epoch number), sliced to
    skip the first `skip_batches * batch_size` samples. Skipped samples
    are never yielded, so the DataLoader never calls __getitem__ for
    them — a real skip, not a load-then-discard. Same epoch + same
    skip_batches always reproduces the same remaining order, so a resume
    lines up exactly with what an uninterrupted run would have done.
    """
    def __init__(self, dataset_len, epoch, batch_size, skip_batches=0, seed=42):
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        indices = torch.randperm(dataset_len, generator=g).tolist()
        self.indices = indices[skip_batches * batch_size:]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def train_one_epoch(
    model, dataset, loss_fn, optimizer, scheduler, device, epoch, total_epochs,
    *, batch_size, num_workers, checkpoint_dir, resume_batch_idx, running, seen, best_eer,
):
    model.train()

    latest_path = os.path.join(checkpoint_dir, "vitcore_audio_latest.pth")
    last_batch_path = os.path.join(checkpoint_dir, "last_batch.txt")
    metrics_path = os.path.join(checkpoint_dir, "accumulated_metrics.pth")

    sampler = ResumableSampler(len(dataset), epoch, batch_size, skip_batches=resume_batch_idx)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    pbar = tqdm(
        loader, desc=f"Epoch {epoch}/{total_epochs} [train]",
        initial=resume_batch_idx, total=resume_batch_idx + len(loader),
    )
    for i, batch in enumerate(pbar):
        batch_idx = resume_batch_idx + i

        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, f1_norm, f2_norm = model(view1, view2)
        losses = loss_fn(logits, f1_norm, f2_norm, labels)
        losses["total"].backward()
        optimizer.step()

        bsz = labels.size(0)
        seen += bsz
        for k in running:
            running[k] += losses[k].item() * bsz

        pbar.set_postfix({"loss": f"{running['total'] / seen:.4f}"})

        if batch_idx % CHECKPOINT_EVERY_N_BATCHES == 0 and batch_idx != 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_eer": best_eer,
                "running": running,
                "seen": seen,
            }, latest_path)
            with open(last_batch_path, "w") as f:
                f.write(str(batch_idx))
            torch.save({"running": running, "seen": seen, "best_eer": best_eer}, metrics_path)

    return running, seen


@torch.inference_mode()
def evaluate(model, loader, device, epoch, total_epochs):
    model.eval()
    all_labels, all_scores, all_preds = [], [], []

    for batch in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [val]"):
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        labels = batch["label"]

        logits, _, _ = model(view1, view2)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.numpy())
        all_scores.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return compute_all(all_labels, all_scores, all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--train_audio_dir", required=True)
    parser.add_argument("--val_protocol", required=True)
    parser.add_argument("--val_audio_dir", required=True)
    parser.add_argument("--preprocessed", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--file_ext", default=".flac")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.preprocessed:
        train_ds = PreprocessedAudioSpoofDataset(args.train_protocol, args.train_audio_dir, train=True)
        val_ds = PreprocessedAudioSpoofDataset(args.val_protocol, args.val_audio_dir, train=False)
    else:
        from datasets import AudioSpoofDataset
        train_ds = AudioSpoofDataset(args.train_protocol, args.train_audio_dir, train=True, file_ext=args.file_ext)
        val_ds = AudioSpoofDataset(args.val_protocol, args.val_audio_dir, train=False, file_ext=args.file_ext)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ViTCoreAudio(num_classes=2).to(device)
    loss_fn = ViTCoreAudioLoss(consistency_weight=args.consistency_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    latest_path = os.path.join(args.checkpoint_dir, "vitcore_audio_latest.pth")
    best_path = os.path.join(args.checkpoint_dir, "vitcore_audio_best.pth")
    last_batch_path = os.path.join(args.checkpoint_dir, "last_batch.txt")
    metrics_path = os.path.join(args.checkpoint_dir, "accumulated_metrics.pth")
    exit_path = os.path.join(args.checkpoint_dir, "vitcore_audio_exit.pth")
    csv_path = os.path.join(args.checkpoint_dir, "vitcore_audio_losses.csv")

    start_epoch = 1
    resume_batch_idx = 0
    running = {"total": 0.0, "classification": 0.0, "consistency": 0.0}
    seen = 0
    best_eer = float("inf")

    if os.path.exists(latest_path):
        print(f"[Resume] Loading model/optimizer/scheduler state from {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 1)
        best_eer = checkpoint.get("best_eer", float("inf"))

    if os.path.exists(last_batch_path):
        with open(last_batch_path, "r") as f:
            resume_batch_idx = int(f.read())
        print(f"[Resume] Resuming epoch {start_epoch} from batch {resume_batch_idx} (no data reloaded for completed batches)")

    if os.path.exists(metrics_path):
        print(f"[Resume] Loading running metrics from {metrics_path}")
        metrics_checkpoint = torch.load(metrics_path, weights_only=False)
        running = metrics_checkpoint.get("running", running)
        seen = metrics_checkpoint.get("seen", 0)
        best_eer = metrics_checkpoint.get("best_eer", best_eer)

    def save_on_exit():
        torch.save({
            "epoch": start_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_eer": best_eer,
            "running": running,
            "seen": seen,
        }, exit_path)
        print(f"[Exit Save] Model saved to {exit_path}")

    atexit.register(save_on_exit)

    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "total_loss", "classification_loss", "consistency_loss",
                              "val_accuracy", "val_auc", "val_eer_pct"])

    for epoch in range(start_epoch, args.epochs + 1):
        if resume_batch_idx == 0:
            running = {"total": 0.0, "classification": 0.0, "consistency": 0.0}
            seen = 0

        running, seen = train_one_epoch(
            model, train_ds, loss_fn, optimizer, scheduler, device, epoch, args.epochs,
            batch_size=args.batch_size, num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir, resume_batch_idx=resume_batch_idx,
            running=running, seen=seen, best_eer=best_eer,
        )
        train_metrics = {k: v / seen for k, v in running.items()}

        val_metrics = evaluate(model, val_loader, device, epoch, args.epochs)
        scheduler.step()
        resume_batch_idx = 0

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['total']:.4f} "
            f"(cls={train_metrics['classification']:.4f}, cons={train_metrics['consistency']:.4f}) | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_eer={val_metrics['eer_pct']:.2f}%"
        )

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_metrics["total"], train_metrics["classification"],
                              train_metrics["consistency"], val_metrics["accuracy"],
                              val_metrics["auc"], val_metrics["eer_pct"]])

        if val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_eer": best_eer,
                "val_metrics": val_metrics,
            }, best_path)
            print(f"  -> new best EER {best_eer * 100:.2f}%, saved to {best_path}")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_eer": best_eer,
        }, latest_path)
        if os.path.exists(last_batch_path):
            os.remove(last_batch_path)
        if os.path.exists(metrics_path):
            os.remove(metrics_path)

    print(f"Training complete. Best val EER: {best_eer * 100:.2f}%")


if __name__ == "__main__":
    main()
