"""PrivacyGuard FL - Core Federated Learning: model, server, client, and training loop."""

import copy
from collections import OrderedDict
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class CNNMnist(nn.Module):
    """Lightweight CNN tailored for MNIST, designed to run on constrained hardware."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Client:
    """Simulates one hospital/client that trains locally and sends model update."""

    def __init__(self, client_id: int, train_loader: DataLoader, device: torch.device,
                 lr: float = 0.01, local_epochs: int = 5):
        self.id = client_id
        self.train_loader = train_loader
        self.device = device
        self.local_epochs = local_epochs
        self.model = CNNMnist().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def set_model_weights(self, weights: OrderedDict):
        self.model.load_state_dict(weights)

    def get_model_weights(self) -> OrderedDict:
        return copy.deepcopy(self.model.state_dict())

    def compute_update(self) -> tuple[OrderedDict, float]:
        """Train locally for `local_epochs` and return the weight delta + average loss."""
        initial_weights = copy.deepcopy(self.model.state_dict())
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for _ in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(data)
                total_samples += len(data)

        delta = OrderedDict()
        final_weights = self.model.state_dict()
        for key in initial_weights:
            delta[key] = final_weights[key] - initial_weights[key]

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return delta, avg_loss


class FederatedServer:
    """Aggregates client updates using Federated Averaging (FedAvg)."""

    def __init__(self, device: torch.device, num_classes: int = 10):
        self.device = device
        self.global_model = CNNMnist(num_classes).to(device)
        self.history: List[Dict] = []

    def get_weights(self) -> OrderedDict:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates: List[tuple[OrderedDict, int]]) -> OrderedDict:
        """FedAvg: weighted average of client weight deltas."""
        aggregated = OrderedDict()
        total_samples = sum(samples for _, samples in client_updates)

        if total_samples == 0:
            return self.get_weights()

        for key in self.global_model.state_dict():
            aggregated[key] = torch.zeros_like(self.global_model.state_dict()[key], dtype=torch.float32)

        for delta, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in aggregated:
                aggregated[key] += delta[key].to(torch.float32) * weight

        return aggregated

    def apply_update(self, delta: OrderedDict):
        current = self.global_model.state_dict()
        for key in current:
            current[key] += delta[key].to(current[key].dtype)
        self.global_model.load_state_dict(current)

    def evaluate(self, test_loader: DataLoader):
        self.global_model.eval()
        correct = 0
        total = 0
        loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss += criterion(output, target).item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)
        return correct / total, loss / total


def train_federated(
    server: FederatedServer,
    clients: List[Client],
    test_loader: DataLoader,
    rounds: int = 20,
    verbose: bool = True,
) -> List[Dict]:
    """Run the full federated training loop."""
    iterator = tqdm(range(rounds), desc="FL Round") if verbose else range(rounds)

    for rnd in iterator:
        global_weights = server.get_weights()
        updates = []
        round_losses = []

        for client in clients:
            client.set_model_weights(global_weights)
            delta, loss = client.compute_update()
            updates.append((delta, len(client.train_loader.dataset)))
            round_losses.append(loss)

        aggregated = server.aggregate(updates)
        server.apply_update(aggregated)

        accuracy, test_loss = server.evaluate(test_loader)
        server.history.append({
            "round": rnd + 1,
            "accuracy": accuracy,
            "test_loss": test_loss,
            "train_loss": np.mean(round_losses),
        })

        if verbose:
            iterator.set_postfix(acc=f"{accuracy:.3f}", tloss=f"{test_loss:.3f}")

    return server.history


import numpy as np
