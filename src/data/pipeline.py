import os
import random
import ssl
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context


class DataPipe:
    """Splits MNIST into non-IID partitions across simulated clients."""

    def __init__(
        self,
        num_clients: int = 4,
        samples_per_client: int = 1500,
        non_iid_degree: float = 0.8,
        batch_size: int = 32,
        seed: int = 42,
        data_dir: str = "./data",
    ):
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.non_iid_degree = non_iid_degree
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def load_full_dataset(self):
        train_set = datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        test_set = datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )
        return train_set, test_set

    def _non_iid_split(self, train_set):
        indices_by_label = defaultdict(list)
        for idx, (_, label) in enumerate(train_set):
            indices_by_label[int(label)].append(idx)

        client_indices = [[] for _ in range(self.num_clients)]
        labels = list(range(10))

        primary_count = int(self.samples_per_client * self.non_iid_degree)
        secondary_count = self.samples_per_client - primary_count

        for c in range(self.num_clients):
            primary_labels = labels[c % 10 : (c % 10) + 2]
            if len(primary_labels) < 2:
                primary_labels = labels[:2]
            primary_labels = [l % 10 for l in primary_labels]
            per_primary = primary_count // len(primary_labels)

            for pl in primary_labels:
                pool = indices_by_label[pl]
                chosen = random.sample(pool, min(per_primary, len(pool)))
                client_indices[c].extend(chosen)

            remaining_labels = [l for l in labels if l not in primary_labels]
            for _ in range(secondary_count):
                rl = random.choice(remaining_labels)
                pool = indices_by_label[rl]
                if pool:
                    client_indices[c].append(random.choice(pool))

        return client_indices

    def build_client_loaders(self):
        train_set, test_set = self.load_full_dataset()
        client_indices = self._non_iid_split(train_set)
        client_loaders = []

        for idx in client_indices:
            subset = Subset(train_set, idx)
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            client_loaders.append(loader)

        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        return client_loaders, test_loader, client_indices

    def compute_label_distribution(self, train_set, client_indices):
        dists = []
        for idx_list in client_indices:
            labels = [int(train_set[i][1]) for i in idx_list]
            _, counts = np.unique(labels, return_counts=True)
            dists.append(counts / counts.sum())
        return dists
