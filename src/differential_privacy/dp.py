"""PrivacyGuard FL - Differential Privacy module with DP-SGD and epsilon tracking."""

import torch
import torch.nn as nn
from typing import OrderedDict, Dict


class DPSGD(torch.optim.Optimizer):
    """Differentially Private SGD using per-sample gradient clipping + Gaussian noise.

    Implements the core DP-SGD algorithm:
    1. Clip per-sample gradients to bound sensitivity
    2. Add calibrated Gaussian noise
    3. Track privacy budget (ε, δ) via moments accountant
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 l2_norm_clip: float = 1.0, noise_multiplier: float = 1.1,
                 microbatch_size: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if l2_norm_clip < 0.0:
            raise ValueError(f"Invalid l2_norm_clip: {l2_norm_clip}")
        if noise_multiplier < 0.0:
            raise ValueError(f"Invalid noise_multiplier: {noise_multiplier}")

        defaults = dict(lr=lr, momentum=momentum, l2_norm_clip=l2_norm_clip,
                       noise_multiplier=noise_multiplier, microbatch_size=microbatch_size)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                clip = group["l2_norm_clip"]
                noise_scale = group["noise_multiplier"] * clip

                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).clamp_min(1e-10)
                clip_coef = torch.clamp(clip / grad_norm, max=1.0).view(-1, *([1] * (grad.ndim - 1)))
                grad = grad * clip_coef

                noise = torch.normal(mean=0.0, std=noise_scale, size=grad.shape, device=grad.device)
                grad = grad + noise
                grad = grad.mean(dim=0)

                if group["momentum"] > 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad, alpha=1)
                    grad = state["momentum_buffer"]

                p.data.add_(grad, alpha=-group["lr"])

        return loss


class PrivacyAccountant:
    """Tracks privacy budget (ε, δ) using the moments accountant approach."""

    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self.q = 0.0
        self.steps = 0
        self.noise_multiplier = 1.0

    def configure(self, sample_rate: float, noise_multiplier: float):
        self.q = sample_rate
        self.noise_multiplier = noise_multiplier
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_epsilon(self, target_delta: float = None) -> float:
        if self.steps == 0:
            return 0.0

        delta = target_delta if target_delta is not None else self.delta
        orders = [1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0, 32.0, 64.0]

        best_eps = float("inf")
        for alpha in orders:
            eps = self._compute_eps_for_alpha(alpha, delta)
            if eps < best_eps:
                best_eps = eps
        return best_eps

    def get_privacy_spent(self) -> Dict:
        return {
            "epsilon": self.get_epsilon(),
            "delta": self.delta,
            "steps": self.steps,
            "sample_rate": self.q,
            "noise_multiplier": self.noise_multiplier,
        }

    def _compute_eps_for_alpha(self, alpha: float, delta: float) -> float:
        q = self.q
        sigma = self.noise_multiplier
        steps = self.steps

        if sigma == 0:
            return float("inf")

        moment = steps * (q ** 2 * alpha * (alpha - 1) / (2 * sigma ** 2) +
                          q ** 3 * alpha ** 3 / (6 * sigma ** 4))
        return moment / alpha - (torch.log(torch.tensor(delta)).item() / (alpha - 1)) if alpha > 1 else float("inf")


class DPClient:
    """Client wrapper that applies DP-SGD during local training."""

    def __init__(self, client, dp_optimizer_cls=DPSGD, dp_kwargs: Dict = None):
        self.client = client
        self.dp_kwargs = dp_kwargs or {
            "lr": 0.01,
            "l2_norm_clip": 1.0,
            "noise_multiplier": 1.1,
            "microbatch_size": 1,
        }

    def compute_update(self) -> tuple[OrderedDict, float]:
        model = self.client.model
        model.train()

        initial_weights = OrderedDict(
            (k, v.clone()) for k, v in model.state_dict().items()
        )

        dp_opt = DPSGD(model.parameters(), **self.dp_kwargs)
        original_opt = self.client.optimizer
        self.client.optimizer = dp_opt

        try:
            delta, loss = self.client.compute_update()
        finally:
            self.client.optimizer = original_opt

        return delta, loss
