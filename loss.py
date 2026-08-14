"""
Loss = classification loss + lambda * consistency loss (L_cons).

Identical formulation to ViT-CORE: cross-entropy on the fused logits,
plus an MSE term forcing the two L2-normalized view embeddings toward
each other. This is modality-agnostic — the loss function doesn't know
or care whether the two "views" it's reconciling came from image crops
or spectrogram transforms, which is exactly why this file needed almost
no changes from the original ViT-CORE implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class ViTCoreAudioLoss(nn.Module):
    def __init__(
        self,
        consistency_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        """
        class_weights: optional per-class weight tensor (shape [num_classes])
            for the classification loss, e.g. from train.py's
            compute_class_weights(). ASVspoof-style protocols are
            typically spoof-dominated, so leaving this unset on an
            imbalanced training set biases the classifier toward the
            majority class.
        label_smoothing: standard regularizer against classifier
            overconfidence, ported from ViT-CORE's training loop.
        """
        super().__init__()
        self.consistency_weight = consistency_weight
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.consistency_loss = nn.MSELoss()

    def forward(self, logits: torch.Tensor, f1_norm: torch.Tensor, f2_norm: torch.Tensor, labels: torch.Tensor):
        cls_loss = self.classification_loss(logits, labels)
        cons_loss = self.consistency_loss(f1_norm, f2_norm)

        total = cls_loss + self.consistency_weight * cons_loss

        return {
            "total": total,
            "classification": cls_loss.detach(),
            "consistency": cons_loss.detach(),
        }
