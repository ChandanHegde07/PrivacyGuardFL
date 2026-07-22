import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class DPClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        epsilon: float = 8.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        learning_rate: float = 0.05,
        local_epochs: int = 3,
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._privacy_spent = 0.0
        self.num_train = len(train_loader.dataset)

        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

    def _compute_noise_multiplier(self) -> float:
        q = 1.0 / self.num_train
        steps = self.local_epochs * max(1, len(self.train_loader))
        sigma_min = 0.1
        sigma_max = 50.0

        for _ in range(50):
            sigma = (sigma_min + sigma_max) / 2
            eps = (
                q
                * self.epsilon
                / sigma
                * np.sqrt(steps * np.log(1.0 / self.delta))
            )  # approximate RDP
            if eps > self.epsilon:
                sigma_min = sigma
            else:
                sigma_max = sigma

        return sigma_max

    def _estimate_privacy_spent(self, steps: int):
        q = 1.0 / self.num_train
        self._privacy_spent += (
            q
            * self.noise_multiplier
            * np.sqrt(steps * np.log(1.0 / self.delta))
        )

    def _clip_gradients(self, parameters, clip_norm: float):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
        total_norm = total_norm**0.5

        clip_coef = min(clip_norm / (total_norm + 1e-6), 1.0)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    def _add_noise(self, parameters):
        for p in parameters:
            if p.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=self.noise_multiplier * self.clip_norm,
                    size=p.grad.shape,
                    device=self.device,
                )
                p.grad.data.add_(noise)

    def train_local(self) -> Dict[str, np.ndarray]:
        self.model.train()
        steps = 0

        for _ in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.model.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                self._clip_gradients(self.model.parameters(), self.clip_norm)
                self._add_noise(self.model.parameters())

                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.data -= self.learning_rate * p.grad

                steps += 1

        self._estimate_privacy_spent(steps)
        return self.get_weights()

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {n: p.cpu().numpy().copy() for n, p in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, np.ndarray]):
        state = self.model.state_dict()
        for name in weights:
            state[name] = torch.from_numpy(weights[name]).to(self.device)
        self.model.load_state_dict(state)

    @property
    def privacy_spent(self) -> float:
        return self._privacy_spent


class DPFedServer:
    def __init__(
        self,
        model: nn.Module,
        clients: List[DPClient],
        test_loader: DataLoader,
        rounds: int = 20,
        fraction_fit: float = 1.0,
        server_noise_std: float = 0.0,
        device: torch.device = None,
    ):
        self.global_model = copy.deepcopy(model)
        self.clients = clients
        self.test_loader = test_loader
        self.rounds = rounds
        self.fraction_fit = fraction_fit
        self.server_noise_std = server_noise_std
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        self.history: Dict[str, List[float]] = {
            "test_accuracy": [],
            "test_loss": [],
            "privacy_spent": [],
        }

    def _get_global_weights(self) -> Dict[str, np.ndarray]:
        return {n: p.cpu().numpy().copy() for n, p in self.global_model.state_dict().items()}

    def _federated_averaging(self, client_weights: List[Dict[str, np.ndarray]]):
        avg = {}
        for key in client_weights[0]:
            stacked = np.stack([cw[key] for cw in client_weights])
            avg[key] = np.mean(stacked, axis=0)

            if self.server_noise_std > 0:
                avg[key] += np.random.normal(
                    0, self.server_noise_std, avg[key].shape
                )

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
        for rnd in tqdm(range(self.rounds), desc="DP-FL Rounds"):
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
            max_eps = max(c.privacy_spent for c in self.clients)

            self.history["test_accuracy"].append(accuracy)
            self.history["test_loss"].append(loss)
            self.history["privacy_spent"].append(max_eps)

            if progress_callback:
                progress_callback(rnd, accuracy, loss, max_eps)

        return self.history
