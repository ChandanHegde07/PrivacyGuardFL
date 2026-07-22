from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
         data_range: float = 1.0) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = img1.detach()
    img2 = img2.detach()

    if img1.dim() == 4:
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)
    if img1.dim() == 3 and img1.shape[0] == 1:
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)

    kernel = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size ** 2)
    kernel_h, kernel_w = kernel.shape[-2:]

    img1 = img1.unsqueeze(0).unsqueeze(0)
    img2 = img2.unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(img1, kernel, padding=kernel_h // 2)
    mu2 = F.conv2d(img2, kernel, padding=kernel_h // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=kernel_h // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=kernel_h // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_h // 2) - mu12

    num = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    return float(ssim_map.mean().item())


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def _infer_label_from_gradient(target_grads: Dict[str, torch.Tensor]) -> int:
    dW_fc2 = target_grads["fc2.bias"]
    return int(torch.argmin(dW_fc2).item())


class GradientInversionAttack:
    """Reconstructs training images from leaked gradients (iDLG + DLG).

    Implements the standard DLG/iDLG attack with per-layer cosine-distances
    as the matching objective. The true label is inferred analytically from
    the sign of the final-layer bias gradient (iDLG).

    Architecture:
      - iDLG closed-form label recovery from ∇b_fc2
      - dummy_x initialized as 0.1 × randn
      - Adam optimizer, ~1000 iterations, cosine + MSE loss on all param grads
      - Total-variation regularizer at 1e-4
      - Per-iteration loss printed to terminal

    References:
      - Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019
      - Zhao et al., "iDLG: Improved Deep Leakage from Gradients", 2020
    """

    def __init__(
        self,
        model: nn.Module,
        weights: Dict[str, np.ndarray],
        grad: Dict[str, np.ndarray],
        tv_weight: float = 1e-4,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tv_weight = tv_weight

        self.original_model = model
        self.attack_model = model.__class__().to(self.device)
        self.attack_model.load_state_dict(
            {n: torch.from_numpy(v).to(self.device) for n, v in weights.items()}
        )
        self.attack_model.eval()

        self.target_grads = {
            n: torch.from_numpy(g).to(self.device) for n, g in grad.items()
        }

    def _compute_match_loss(
        self,
        dummy_x: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        self.attack_model.zero_grad()
        pred = self.attack_model(dummy_x)
        ce = F.cross_entropy(pred, label)

        dummy_grads = torch.autograd.grad(
            ce, self.attack_model.parameters(), create_graph=True
        )

        num_layers = len(dummy_grads)
        param_names = [n for n, _ in self.attack_model.named_parameters()]

        loss_parts = []
        cos_sim_total = 0.0

        for i, (name, dummy) in enumerate(zip(param_names, dummy_grads)):
            if name not in self.target_grads:
                continue
            target = self.target_grads[name]
            d_flat = dummy.view(-1)
            t_flat = target.view(-1)

            dot = (d_flat * t_flat).sum()
            denom = d_flat.norm() * t_flat.norm() + 1e-10
            cos = dot / denom

            loss_parts.append(1.0 - cos)
            cos_sim_total += float(cos.item())

        n_valid = max(len(loss_parts), 1)
        match_loss = sum(loss_parts) / n_valid
        mean_cos = cos_sim_total / n_valid

        return match_loss, mean_cos

    def attack(
        self,
        batch_size: int = 1,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        steps: int = 1000,
        learning_rate: float = 0.1,
        label_hint: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, int, Dict[str, list]]:
        if label_hint is not None:
            true_label = label_hint
            if verbose:
                print(f"  [iDLG] Using provided label hint: {true_label}")
        else:
            true_label = _infer_label_from_gradient(self.target_grads)
            if verbose:
                print(f"  [iDLG] Inferred label from ∇b_fc2: {true_label}")

        label = torch.tensor([true_label] * batch_size, device=self.device, dtype=torch.long)

        dummy_x = 0.1 * torch.randn(batch_size, *input_shape, device=self.device)
        dummy_x.requires_grad_(True)

        optimizer = optim.Adam([dummy_x], lr=learning_rate)
        history = {"loss": [], "cos_sim": []}
        best_img = None
        best_loss = float("inf")

        for it in range(1, steps + 1):
            optimizer.zero_grad()
            self.attack_model.zero_grad()

            match_loss, cos_sim = self._compute_match_loss(dummy_x, label)
            tv = _total_variation(dummy_x)
            total_loss = match_loss + self.tv_weight * tv

            total_loss.backward()
            optimizer.step()
            dummy_x.data.clamp_(0, 1)

            history["loss"].append(float(match_loss.item()))
            history["cos_sim"].append(float(cos_sim))

            if match_loss.item() < best_loss:
                best_loss = match_loss.item()
                best_img = dummy_x.detach().clone()

            if verbose and it % 200 == 0:
                print(f"    iter {it:4d}/{steps}  loss={match_loss.item():.8f}  "
                      f"cos_sim={cos_sim:.4f}  TV={tv.item():.6f}")

        if verbose:
            final_loss = history["loss"][-1]
            final_cos = history["cos_sim"][-1]
            print(f"    Final => iters={steps}  loss={final_loss:.8f}  "
                  f"cos_sim={final_cos:.4f} (best loss={best_loss:.8f})")

        return best_img if best_img is not None else dummy_x.detach(), true_label, history

    @staticmethod
    def evaluate_reconstruction_mse(
        original: torch.Tensor, reconstructed: torch.Tensor
    ) -> float:
        original = original.to(reconstructed.device)
        return F.mse_loss(reconstructed, original).item()

    @staticmethod
    def evaluate_ssim(
        original: torch.Tensor, reconstructed: torch.Tensor, window_size: int = 11
    ) -> float:
        return ssim(original, reconstructed, window_size=window_size)
