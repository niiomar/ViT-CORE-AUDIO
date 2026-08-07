"""
Standalone evaluation: load a trained checkpoint, run it against any
ASVspoof-protocol-formatted test set, report accuracy/AUC/EER.

Kept separate from train.py (rather than only exposing evaluate() there)
specifically so a checkpoint can be scored against a DIFFERENT dataset
than it was trained on — e.g. train on ASVspoof2019 LA, evaluate on
In-the-Wild — which is the standard cross-dataset generalization check
this field expects (a model that only reports low EER on its own
training distribution's held-out split is a much weaker claim than one
that holds up cross-dataset).
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import AudioSpoofDataset
from metrics import compute_all
from model import ViTCoreAudio


@torch.inference_mode()
def run_evaluation(model, loader, device):
    model.eval()
    all_labels, all_scores, all_preds, all_filenames = [], [], [], []

    for batch in loader:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)

        logits, _, _ = model(view1, view2)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(batch["label"].numpy())
        all_scores.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_filenames.extend(batch["filename"])

    metrics = compute_all(all_labels, all_scores, all_preds)
    per_file_scores = list(zip(all_filenames, all_scores, all_labels))
    return metrics, per_file_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", default=None, help="optional path to dump per-file scores + summary metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTCoreAudio(num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"training-time val EER {ckpt.get('val_eer', float('nan')) * 100:.2f}%")

    ds = AudioSpoofDataset(args.protocol, args.audio_dir, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metrics, per_file_scores = run_evaluation(model, loader, device)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC:      {metrics['auc']:.4f}")
    print(f"EER:      {metrics['eer_pct']:.2f}%  (threshold={metrics['eer_threshold']:.4f})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({
                "metrics": metrics,
                "per_file": [
                    {"filename": fn, "spoof_score": float(s), "label": int(l)}
                    for fn, s, l in per_file_scores
                ],
            }, f, indent=2)
        print(f"\nDetailed results written to {args.output_json}")


if __name__ == "__main__":
    main()
