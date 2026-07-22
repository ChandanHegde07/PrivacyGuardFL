import copy
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Client:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        learning_rate: float = 0.01,
        local_epochs: int = 3,
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_local(self) -> Dict[str, np.ndarray]:
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return self.get_weights()

    def get_weights(self) -> Dict[str, np.ndarray]:
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.cpu().numpy().copy()
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]):
        state = self.model.state_dict()
        for name in weights:
            state[name] = torch.from_numpy(weights[name]).to(self.device)
        self.model.load_state_dict(state)

    def get_gradients(self) -> Dict[str, np.ndarray]:
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.cpu().numpy().copy()
        return grads


class FedServer:
    def __init__(
        self,
        model: nn.Module,
        clients: List[Client],
        test_loader: DataLoader,
        rounds: int = 20,
        fraction_fit: float = 1.0,
        device: torch.device = None,
    ):
        self.global_model = copy.deepcopy(model)
        self.clients = clients
        self.test_loader = test_loader
        self.rounds = rounds
        self.fraction_fit = fraction_fit
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "test_accuracy": [],
            "test_loss": [],
        }

    def _get_global_weights(self) -> Dict[str, np.ndarray]:
        return {n: p.cpu().numpy().copy() for n, p in self.global_model.state_dict().items()}

    def _federated_averaging(self, client_weights: List[Dict[str, np.ndarray]]):
        avg = {}
        for key in client_weights[0]:
            avg[key] = np.mean([cw[key] for cw in client_weights], axis=0)

        state = self.global_model.state_dict()
        for key in avg:
            state[key] = torch.from_numpy(avg[key]).to(self.device)
        self.global_model.load_state_dict(state)

    def _evaluate(self) -> Tuple[float, float]:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct = 0.0, 0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += data.size(0)

        accuracy = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return accuracy, avg_loss

    def fit(self, progress_callback=None) -> Dict[str, List[float]]:
        for rnd in tqdm(range(self.rounds), desc="FL Rounds"):
            n_selected = max(1, int(self.fraction_fit * len(self.clients)))
            selected = np.random.choice(self.clients, n_selected, replace=False)

            client_weights = []
            global_weights = self._get_global_weights()

            for client in selected:
                client.set_weights(global_weights)
                w = client.train_local()
                client_weights.append(w)

            self._federated_averaging(client_weights)

            accuracy, loss = self._evaluate()
            self.history["test_accuracy"].append(accuracy)
            self.history["test_loss"].append(loss)

            if progress_callback:
                progress_callback(rnd, accuracy, loss)

        return self.history
