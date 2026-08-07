"""
Training loop for ViT-CORE-Audio. Structurally identical to ViT-CORE's
train.py — same optimizer/scheduler pattern, same checkpoint-on-best-EER
logic (adapted from checkpoint-on-best-val-accuracy, since EER is this
domain's actual model-selection metric).
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import AudioSpoofDataset
from loss import ViTCoreAudioLoss
from metrics import compute_all
from model import ViTCoreAudio


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running = {"total": 0.0, "classification": 0.0, "consistency": 0.0}

    for batch in loader:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, f1_norm, f2_norm = model(view1, view2)
        losses = loss_fn(logits, f1_norm, f2_norm, labels)
        losses["total"].backward()
        optimizer.step()

        for k in running:
            running[k] += losses[k].item() * labels.size(0)

    n = len(loader.dataset)
    return {k: v / n for k, v in running.items()}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_scores, all_preds = [], [], []

    for batch in loader:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        labels = batch["label"]

        logits, _, _ = model(view1, view2)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(spoof)
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = AudioSpoofDataset(args.train_protocol, args.train_audio_dir, train=True)
    val_ds = AudioSpoofDataset(args.val_protocol, args.val_audio_dir, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ViTCoreAudio(num_classes=2).to(device)
    loss_fn = ViTCoreAudioLoss(consistency_weight=args.consistency_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_eer = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['total']:.4f} "
            f"(cls={train_metrics['classification']:.4f}, cons={train_metrics['consistency']:.4f}) | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_eer={val_metrics['eer_pct']:.2f}%"
        )

        if val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            ckpt_path = os.path.join(args.checkpoint_dir, "vitcore_audio_best.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_eer": best_eer,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  -> new best EER {best_eer * 100:.2f}%, saved to {ckpt_path}")

    print(f"Training complete. Best val EER: {best_eer * 100:.2f}%")


if __name__ == "__main__":
    main()
