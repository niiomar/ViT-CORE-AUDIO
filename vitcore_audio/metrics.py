"""
Metrics for audio anti-spoofing evaluation.

accuracy() and auc() mirror ViT-CORE's metrics.py directly. eer() is new
— Equal Error Rate is the primary metric every ASVspoof challenge and
essentially all published audio anti-spoofing work reports, and has no
equivalent need in ViT-CORE's vision-domain metrics.py. Omitting it here
would make results incomparable to the field's actual benchmarks.

EER = the error rate at the threshold where False Positive Rate (real
audio wrongly flagged as fake) equals False Negative Rate (fake audio
wrongly passed as real). Lower is better; state-of-the-art ASVspoof
2019 LA systems report EER in the 1-5% range.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score, roc_curve


def accuracy(y_true: npt.ArrayLike, y_pred_labels: npt.ArrayLike) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred_labels)))

def auc(y_true: npt.ArrayLike, y_scores: npt.ArrayLike) -> float:
    """y_scores: probability/logit of the POSITIVE (spoof) class."""
    return float(roc_auc_score(y_true, y_scores))

def eer(y_true: npt.ArrayLike, y_scores: npt.ArrayLike) -> tuple[float, float]:
    """
    Returns (eer, threshold) where eer is a fraction in [0, 1] (multiply
    by 100 for the conventional percentage reporting) and threshold is
    the score cutoff at which it occurs.

    y_true:   0 = bonafide (real), 1 = spoof (fake) — matches
              datasets.py's LABEL_MAP convention.
    y_scores: model's predicted probability of the spoof (positive) class.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # EER is where fpr and fnr curves cross. They rarely land on the
    # exact same threshold index, so find the threshold minimizing
    # |fpr - fnr| and take the average of the two rates there as the
    # standard EER estimate (this is the conventional approximation
    # used by the ASVspoof baseline scoring scripts).
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer_value = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]

    return float(eer_value), float(eer_threshold)

def compute_all(y_true: npt.ArrayLike, y_scores: npt.ArrayLike, y_pred_labels: npt.ArrayLike) -> dict:
    eer_value, eer_threshold = eer(y_true, y_scores)
    return {
        "accuracy": accuracy(y_true, y_pred_labels),
        "auc": auc(y_true, y_scores),
        "eer": eer_value,
        "eer_pct": eer_value * 100,
        "eer_threshold": eer_threshold,
    }
