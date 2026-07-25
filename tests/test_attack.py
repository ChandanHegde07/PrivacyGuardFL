"""Tests for gradient inversion attack: runs without crash, correct shapes."""
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pytest
import torch

from src.attacks.gradient_inversion import GradientInversionAttack, ssim, _infer_label_from_gradient
from src.core.federated import MNISTModel


@pytest.fixture
def gradients():
    model = MNISTModel()
    x = torch.randn(1, 1, 28, 28)
    y = torch.tensor([3])
    model.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    weights = {n: p.cpu().numpy().copy() for n, p in model.state_dict().items()}
    grads = {n: p.grad.cpu().numpy().copy() for n, p in model.named_parameters() if p.grad is not None}
    return model, weights, grads, x, y


def test_ssim_same():
    a = torch.randn(1, 28, 28)
    s = ssim(a, a)
    assert abs(s - 1.0) < 0.001


def test_ssim_random():
    a = torch.randn(1, 28, 28)
    b = torch.randn(1, 28, 28)
    s = ssim(a, b)
    assert -0.05 <= s < 1.0


def test_infer_label():
    model = MNISTModel()
    x = torch.randn(1, 1, 28, 28)
    y = torch.tensor([7])
    model.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    label = _infer_label_from_gradient(grads)
    assert label == 7


def test_attack_runs_and_returns_shapes(gradients):
    model, weights, grads, x, y = gradients
    attack = GradientInversionAttack(model, weights, grads)
    recon, inferred_label, history = attack.attack(batch_size=1, steps=10, learning_rate=0.1)
    assert recon.shape == (1, 1, 28, 28)
    assert isinstance(inferred_label, int)
    assert "loss" in history
    assert len(history["loss"]) == 10


def test_mse_eval(gradients):
    model, weights, grads, x, y = gradients
    mse = GradientInversionAttack.evaluate_reconstruction_mse(x, x)
    assert mse == 0.0
