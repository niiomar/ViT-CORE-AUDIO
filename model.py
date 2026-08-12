"""
ViT-CORE-Audio model: identical architecture to ViT-CORE, just fed
spectrogram views instead of image crops. The four-step pipeline from
the ViT-CORE dissertation applies unchanged:

  1. Parallel Augmentation  -> RaAug(mel), DFDC_Selim(cqt)   [datasets.py]
  2. Shared Encoder         -> both views through ONE ViT-S/16   [here]
  3. Feature Embedding      -> L2-normalize each view's embedding [here]
  4. Consistency Constraint -> MSE(f~1, f~2) computed in loss.py

No new architecture is introduced — reusing ViT-S/16 unmodified is the
entire point of resizing both spectral views to 224x224x3 in
audio_preprocessing.py.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import vit_small_patch16_224  # type: ignore[attr-defined]  # registered dynamically by timm

class ViTCoreAudio(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        # Shared encoder — the SAME weights process both views, which is
        # what makes the consistency loss meaningful: if the two views
        # went through separate encoders, forcing their outputs to agree
        # would just be learning two encoders that happen to output
        # similar numbers, not a shared, view-invariant representation.
        self.encoder = vit_small_patch16_224(pretrained=pretrained, num_classes=0)
        embed_dim = self.encoder.embed_dim

        self.classifier = nn.Linear(embed_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (pre-normalization) embedding for one view."""
        return self.encoder(x)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor):
        """
        Returns:
            logits:            classification output, from the AVERAGE
                                of the two views' embeddings (so
                                single-view inference is also supported)
            f1_norm, f2_norm:   L2-normalized embeddings per view,
                                consumed by loss.py's consistency term
        """
        f1 = self.encode(view1)
        f2 = self.encode(view2)

        f1_norm = F.normalize(f1, p=2, dim=-1)
        f2_norm = F.normalize(f2, p=2, dim=-1)

        fused = (f1 + f2) / 2.0
        logits = self.classifier(fused)

        return logits, f1_norm, f2_norm

    def forward_single(self, view: torch.Tensor) -> torch.Tensor:
        """Single-view inference path — e.g. if only a mel-spectrogram
        is available at deploy time and computing a second CQT view
        isn't worth the latency. Trained jointly, usable independently."""
        f = self.encode(view)
        return self.classifier(f)

def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decay/no-decay groups (no weight decay on biases or 1-D norm params) —
    standard practice for transformer fine-tuning, ported from ViT-CORE's model_utils.py."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class ModelEma:
    """Exponential moving average of a model's floating-point parameters and buffers.

    Ported from ViT-CORE's model_utils.py. Validation/checkpointing during
    training prefers these shadow weights over the raw model — they're
    typically less noisy than the last few SGD/AdamW steps and tend to
    generalize slightly better, at the cost of one extra full-model copy
    in memory.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def to(self, device: torch.device) -> ModelEma:
        self.module.to(device)
        return self

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_sd = self.module.state_dict()
        for k, v in model.state_dict().items():
            ema_v = ema_sd[k]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                ema_v.copy_(v)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.module.load_state_dict(state_dict)
