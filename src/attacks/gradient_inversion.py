from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class GradientInversionAttack:
    """Reconstructs training images from leaked gradients.

    Implements Deep Leakage from Gradients (DLG) style attack: optimizes
    dummy inputs and labels to match the target gradients.

    Reference: Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019.
    """

    def __init__(
        self,
        model: nn.Module,
        weights: Dict[str, np.ndarray],
        grad: Dict[str, np.ndarray],
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.original_model = model
        self.attack_model = model.__class__().to(self.device)
        self.attack_model.load_state_dict(
            {n: torch.from_numpy(v).to(self.device) for n, v in weights.items()}
        )
        self.attack_model.eval()

        self.target_grads = {
            n: torch.from_numpy(g).to(self.device) for n, g in grad.items()
        }

    def _gradient_similarity(
        self,
        dummy_grads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = 0.0
        for name in self.target_grads:
            target = self.target_grads[name]
            dummy = dummy_grads[name]
            total += F.cosine_similarity(
                dummy.view(-1), target.view(-1), dim=0
            ).mean()
        return total / max(len(self.target_grads), 1)

    def _compute_gradient_loss(
        self,
        dummy_data: torch.Tensor,
        label: torch.Tensor,
        retain_graph: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        self.attack_model.zero_grad()
        pred = self.attack_model(dummy_data)
        loss = F.cross_entropy(pred, label)
        loss.backward(retain_graph=retain_graph)

        dummy_grads = {}
        for name, param in self.attack_model.named_parameters():
            if param.grad is not None:
                dummy_grads[name] = param.grad.clone()

        similarity = self._gradient_similarity(dummy_grads)
        grad_loss = 1.0 - similarity
        return grad_loss, similarity.item()

    def attack(
        self,
        batch_size: int = 1,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        steps: int = 300,
        learning_rate: float = 0.1,
        label_hint: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        dummy_data = torch.randn(batch_size, *input_shape, device=self.device, requires_grad=True)

        if label_hint is not None:
            dummy_label = torch.tensor([label_hint] * batch_size, device=self.device, dtype=torch.long)
            label_tensor = dummy_label
        else:
            dummy_label_param = torch.randn(batch_size, 10, device=self.device, requires_grad=True)
            label_tensor = dummy_label_param
            label = None

        data_optimizer = optim.Adam([dummy_data], lr=learning_rate)
        label_optimizer = None if label_hint is not None else optim.Adam([label_tensor], lr=learning_rate * 0.5)

        history = {"loss": [], "sim": []}

        for step in range(steps):
            data_optimizer.zero_grad()
            self.attack_model.zero_grad()

            if label_optimizer:
                label_optimizer.zero_grad()
                label = label_tensor.argmax(dim=1)

            current_label = label_tensor if label_hint is not None else label

            pred = self.attack_model(dummy_data)
            ce_loss = F.cross_entropy(pred, current_label)

            dummy_grads = torch.autograd.grad(
                ce_loss, self.attack_model.parameters(),
                create_graph=True, retain_graph=True
            )

            param_names = [n for n, p in self.attack_model.named_parameters()]
            grads_dict = dict(zip(param_names, dummy_grads))

            sim = self._gradient_similarity_from_tensors(grads_dict)
            grad_loss = 1.0 - sim

            history["loss"].append(grad_loss.item())
            history["sim"].append(sim.item())

            grad_loss.backward()

            data_optimizer.step()
            if label_optimizer:
                label_optimizer.step()

            dummy_data.data.clamp_(0, 1)

        if label_hint is None:
            final_label = label_tensor.detach().argmax(dim=1)
        else:
            final_label = dummy_label

        return dummy_data.detach(), final_label, history

    def _gradient_similarity_from_tensors(
        self, dummy_grads: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        count = 0
        for name in dummy_grads:
            if name in self.target_grads:
                target = self.target_grads[name]
                dummy = dummy_grads[name]
                total += F.cosine_similarity(
                    dummy.view(-1), target.view(-1), dim=0
                ).mean()
                count += 1
        return total / max(count, 1)

    @staticmethod
    def evaluate_reconstruction_mse(
        original: torch.Tensor, reconstructed: torch.Tensor
    ) -> float:
        original = original.to(reconstructed.device)
        return F.mse_loss(reconstructed, original).item()

    @staticmethod
    def evaluate_psnr(
        original: torch.Tensor, reconstructed: torch.Tensor
    ) -> float:
        original = original.to(reconstructed.device)
        mse = F.mse_loss(reconstructed, original).item()
        if mse == 0:
            return float("inf")
        return 20 * np.log10(1.0) - 10 * np.log10(mse)
