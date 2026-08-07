import torch

from loss import ViTCoreAudioLoss
from model import ViTCoreAudio


def test_backward_produces_flowing_gradients():
    model = ViTCoreAudio(num_classes=2, pretrained=False)
    loss_fn = ViTCoreAudioLoss(consistency_weight=0.5)

    view1 = torch.randn(2, 3, 224, 224)
    view2 = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    logits, f1_norm, f2_norm = model(view1, view2)
    losses = loss_fn(logits, f1_norm, f2_norm, labels)
    losses["total"].backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_total_loss_equals_classification_plus_weighted_consistency():
    model = ViTCoreAudio(num_classes=2, pretrained=False)
    consistency_weight = 0.7
    loss_fn = ViTCoreAudioLoss(consistency_weight=consistency_weight)

    view1 = torch.randn(2, 3, 224, 224)
    view2 = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    logits, f1_norm, f2_norm = model(view1, view2)
    losses = loss_fn(logits, f1_norm, f2_norm, labels)

    expected = losses["classification"] + consistency_weight * losses["consistency"]
    assert torch.allclose(losses["total"], expected, atol=1e-6)


def test_class_weights_change_the_classification_loss():
    model = ViTCoreAudio(num_classes=2, pretrained=False)
    view1 = torch.randn(4, 3, 224, 224)
    view2 = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 0, 0, 1])

    with torch.no_grad():
        logits, f1_norm, f2_norm = model(view1, view2)

    unweighted = ViTCoreAudioLoss(consistency_weight=0.5)
    weighted = ViTCoreAudioLoss(consistency_weight=0.5, class_weights=torch.tensor([1.0, 5.0]))

    loss_unweighted = unweighted(logits, f1_norm, f2_norm, labels)["classification"]
    loss_weighted = weighted(logits, f1_norm, f2_norm, labels)["classification"]

    assert not torch.allclose(loss_unweighted, loss_weighted)
