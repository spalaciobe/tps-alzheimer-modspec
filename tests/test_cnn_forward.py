"""Verifica forward, backward y shape-equivalence de la CNN."""

from __future__ import annotations

import torch

from src.models.cnn import ModSpecCNN


def test_forward_shape():
    model = ModSpecCNN(in_channels=19, num_classes=2)
    x = torch.randn(4, 19, 45, 45)
    y = model(x)
    assert y.shape == (4, 2)


def test_backward_runs():
    model = ModSpecCNN(in_channels=19, num_classes=2)
    x = torch.randn(2, 19, 45, 45)
    target = torch.tensor([0, 1])
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_gradcam_layer_exists():
    model = ModSpecCNN()
    layer = model.gradcam_target_layer
    assert isinstance(layer, torch.nn.Conv2d)
