import torch
import torch.nn as nn

from model import ModelEma, ViTCoreAudio, build_param_groups


def _make_model() -> ViTCoreAudio:
    # pretrained=False: tests must not depend on network access to
    # download ImageNet weights.
    return ViTCoreAudio(num_classes=2, pretrained=False)


def _tiny_module() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


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


def test_build_param_groups_splits_bias_and_norm_params_without_decay():
    model = _make_model()
    groups = build_param_groups(model, weight_decay=0.05)

    assert len(groups) == 2
    decay_group, no_decay_group = groups
    assert decay_group["weight_decay"] == 0.05
    assert no_decay_group["weight_decay"] == 0.0

    for p in no_decay_group["params"]:
        assert p.ndim <= 1

    total_from_groups = sum(p.numel() for g in groups for p in g["params"])
    total_from_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_from_groups == total_from_model


def test_model_ema_starts_as_exact_copy_and_freezes_gradients():
    model = _tiny_module()
    ema = ModelEma(model, decay=0.9)

    for p in ema.module.parameters():
        assert not p.requires_grad

    for (name, p), (ema_name, ema_p) in zip(model.state_dict().items(), ema.state_dict().items(), strict=True):
        assert name == ema_name
        assert torch.equal(p, ema_p)


def test_model_ema_update_applies_correct_decay_formula():
    model = _tiny_module()
    decay = 0.9
    ema = ModelEma(model, decay=decay)
    old_ema_params = [p.clone() for p in ema.module.parameters()]

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    ema.update(model)

    for old_ema, new_model_p, new_ema_p in zip(
        old_ema_params, model.parameters(), ema.module.parameters(), strict=True
    ):
        expected = decay * old_ema + (1 - decay) * new_model_p
        assert torch.allclose(new_ema_p, expected, atol=1e-6)


def test_model_ema_state_dict_round_trip():
    model = _tiny_module()
    ema = ModelEma(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)

    fresh_ema = ModelEma(_tiny_module(), decay=0.9)
    fresh_ema.load_state_dict(ema.state_dict())

    for p_a, p_b in zip(ema.module.parameters(), fresh_ema.module.parameters(), strict=True):
        assert torch.equal(p_a, p_b)
