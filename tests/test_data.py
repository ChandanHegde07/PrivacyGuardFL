"""Tests for data pipeline: non-IID splitting, DataLoader shapes, label skew."""
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pytest

from src.data.pipeline import DataPipe

BATCH_SIZE = 32


@pytest.fixture
def pipe():
    return DataPipe(
        num_clients=4,
        samples_per_client=500,
        non_iid_degree=0.8,
        batch_size=BATCH_SIZE,
    )


def test_client_count(pipe):
    loaders, test_loader, indices = pipe.build_client_loaders()
    assert len(loaders) == 4


def test_samples_per_client(pipe):
    _, _, indices = pipe.build_client_loaders()
    for idx_list in indices:
        assert len(idx_list) == 500


def test_test_loader_shape(pipe):
    _, test_loader, _ = pipe.build_client_loaders()
    data, target = next(iter(test_loader))
    assert data.shape == (BATCH_SIZE, 1, 28, 28)
    assert target.shape == (BATCH_SIZE,)


def test_non_iid_skew(pipe):
    train_set, _ = pipe.load_full_dataset()
    _, _, indices = pipe.build_client_loaders()
    dists = pipe.compute_label_distribution(train_set, indices)
    assert len(dists) == 4
    for dist in dists:
        assert len(dist) == 10
        assert abs(dist.sum() - 1.0) < 1e-6


def test_label_distribution_extreme():
    """Degree=0 is uniform, degree=1 is extreme skew."""
    for degree, max_class in [(0.0, 10), (1.0, 2)]:
        p = DataPipe(num_clients=4, samples_per_client=100, non_iid_degree=degree)
        train_set, _ = p.load_full_dataset()
        _, _, indices = p.build_client_loaders()
        for idx_list in indices:
            labels = [int(train_set[i][1]) for i in idx_list]
            unique = len(set(labels))
            assert unique <= max_class, f"degree={degree}: expected ≤{max_class} classes, got {unique}"
