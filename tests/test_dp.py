"""Tests for DP-SGD client: clipping, noise, and epsilon accounting."""
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.federated import MNISTModel
from src.differential_privacy.dp_client import DPClient


def _dummy_loader(batch_size=16, n=500):
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_dp_client_init():
    loader = _dummy_loader()
    client = DPClient(
        0, MNISTModel(), loader,
        epsilon=8.0, delta=1e-5, clip_norm=1.0,
        local_epochs=1, total_rounds=1,
    )
    assert client.noise_multiplier > 0
    assert 0 < client.sample_rate < 1


def test_dp_train_updates_eps():
    loader = _dummy_loader(batch_size=32, n=500)
    client = DPClient(
        0, MNISTModel(), loader,
        epsilon=8.0, delta=1e-5, clip_norm=1.0,
        local_epochs=1, total_rounds=1,
    )
    eps0 = client.privacy_spent
    client.train_local()
    eps1 = client.privacy_spent
    assert eps1 >= eps0
    assert eps1 > 0


def test_clip_norm_applies():
    loader = _dummy_loader(batch_size=1, n=1)
    model = MNISTModel()
    client = DPClient(0, model, loader, epsilon=128.0, clip_norm=0.001, local_epochs=1, total_rounds=1)
    x, y = next(iter(loader))
    client.model.zero_grad()
    out = client.model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    client._clip_gradients(client.model.parameters(), 0.001)
    total_norm_sq = 0.0
    for p in client.model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.norm().item() ** 2
    total_norm = total_norm_sq ** 0.5
    assert total_norm <= 0.001 + 1e-6, f"expected ≤0.001, got {total_norm}"


def test_noise_added():
    loader = _dummy_loader(batch_size=32, n=500)
    client = DPClient(0, MNISTModel(), loader, epsilon=4.0, clip_norm=1.0, local_epochs=1, total_rounds=1)
    x, y = next(iter(loader))
    client.model.zero_grad()
    out = client.model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grad_before = {n: p.grad.clone() for n, p in client.model.named_parameters() if p.grad is not None}
    client._clip_gradients(client.model.parameters(), 1.0)
    client._add_noise(client.model.parameters())
    for n, p in client.model.named_parameters():
        if p.grad is not None:
            assert not torch.allclose(grad_before[n], p.grad), f"noise not added to {n}"
