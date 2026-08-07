import torch

from model import ViTCoreAudio


def _make_model() -> ViTCoreAudio:
    # pretrained=False: tests must not depend on network access to
    # download ImageNet weights.
    return ViTCoreAudio(num_classes=2, pretrained=False)


def test_forward_returns_unit_norm_embeddings():
    model = _make_model()
    model.eval()
    view1 = torch.randn(2, 3, 224, 224)
    view2 = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits, f1_norm, f2_norm = model(view1, view2)

    assert logits.shape == (2, 2)
    for f in (f1_norm, f2_norm):
        norms = f.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_forward_single_matches_expected_shape():
    model = _make_model()
    model.eval()
    view = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        logits = model.forward_single(view)

    assert logits.shape == (1, 2)


def test_forward_logits_are_average_of_view_embeddings_through_shared_classifier():
    """The shared encoder + shared classifier structure is the whole
    point of the architecture — verify fused logits actually come from
    averaging both views' raw (pre-normalization) embeddings."""
    model = _make_model()
    model.eval()
    view1 = torch.randn(1, 3, 224, 224)
    view2 = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        logits, _, _ = model(view1, view2)
        f1 = model.encode(view1)
        f2 = model.encode(view2)
        expected_logits = model.classifier((f1 + f2) / 2.0)

    assert torch.allclose(logits, expected_logits, atol=1e-5)
