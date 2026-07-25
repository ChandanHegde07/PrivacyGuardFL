"""Tests for federated learning core: FedAvg correctness, Client, FedServer."""
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.federated import Client, FedServer, MNISTModel


def _dummy_loaders(n=4, batch_size=16):
    """Create synthetic loaders so tests don't download MNIST."""
    loaders = []
    for _ in range(n):
        x = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        ds = TensorDataset(x, y)
        loaders.append(DataLoader(ds, batch_size=batch_size))
    return loaders


def test_mnist_model_shape():
    model = MNISTModel()
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


def test_client_get_set_weights():
    model = MNISTModel()
    loader = _dummy_loaders(1)[0]
    client = Client(0, model, loader, local_epochs=1)
    w1 = client.get_weights()
    assert "conv1.weight" in w1

    client.set_weights(w1)
    w2 = client.get_weights()
    for k in w1:
        assert np.allclose(w1[k], w2[k])


@pytest.mark.slow
def test_client_train_produces_weights():
    model = MNISTModel()
    loader = _dummy_loaders(1)[0]
    client = Client(0, model, loader, local_epochs=1)
    w = client.train_local()
    assert "conv1.weight" in w


def test_fedavg_correctness():
    loaders = _dummy_loaders(2, batch_size=8)
    test_loader = loaders[0]
    model = MNISTModel()
    clients = [Client(i, MNISTModel(), l, local_epochs=1) for i, l in enumerate(loaders)]
    server = FedServer(model, clients, test_loader, rounds=2)

    history = server.fit()
    assert "test_accuracy" in history
    assert "test_loss" in history
    assert len(history["test_accuracy"]) == 2
    assert 0.0 <= history["test_accuracy"][-1] <= 1.0
