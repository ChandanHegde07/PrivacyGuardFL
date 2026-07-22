from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..core.federated import MNISTModel


def plot_training_history(
    history: Dict[str, List[float]],
    dp_history: Optional[Dict[str, List[float]]] = None,
    save_path: str = "training_history.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["test_accuracy"], label="No DP", color="steelblue", linewidth=2)
    if dp_history:
        axes[0].plot(dp_history["test_accuracy"], label="With DP", color="darkorange", linewidth=2)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Test Accuracy over Rounds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["test_loss"], label="No DP", color="steelblue", linewidth=2)
    if dp_history:
        axes[1].plot(dp_history["test_loss"], label="With DP", color="darkorange", linewidth=2)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Test Loss over Rounds")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if dp_history and "privacy_spent" in dp_history:
        axes[2].plot(dp_history["privacy_spent"], color="crimson", linewidth=2)
        axes[2].axhline(y=8, color="gray", linestyle="--", label="Target ε=8")
        axes[2].set_xlabel("Round")
        axes[2].set_ylabel("ε (Privacy Budget)")
        axes[2].set_title("Privacy Budget Spent")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No DP tracking available", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title("Privacy Budget")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_prediction(probabilities: np.ndarray, save_path: str = "prediction.png"):
    digits = list(range(10))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(digits, probabilities, color=["steelblue" if i != np.argmax(probabilities)
                  else "darkorange" for i in digits])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title(f"Prediction: {np.argmax(probabilities)} (Confidence: {probabilities.max():.3f})")
    ax.set_xticks(digits)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_attack_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: str = "attack_comparison.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    orig = original.squeeze().cpu().numpy()
    recon = reconstructed.squeeze().cpu().numpy()

    if orig.ndim == 2:
        axes[0].imshow(orig, cmap="gray")
        axes[1].imshow(recon, cmap="gray")
    else:
        axes[0].imshow(orig[0], cmap="gray")
        axes[1].imshow(recon[0], cmap="gray")

    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    mse = float(torch.nn.functional.mse_loss(reconstructed, original))
    plt.suptitle(f"Gradient Inversion Attack  |  Reconstruction MSE: {mse:.4f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_label_distribution(
    distributions: List[np.ndarray],
    save_path: str = "label_distribution.png",
):
    n_clients = len(distributions)
    fig, axes = plt.subplots(1, n_clients, figsize=(4 * n_clients, 4))

    if n_clients == 1:
        axes = [axes]

    for i, dist in enumerate(distributions):
        axes[i].bar(range(10), dist, color="steelblue")
        axes[i].set_xlabel("Digit")
        axes[i].set_ylabel("Proportion")
        axes[i].set_title(f"Client {i+1}")
        axes[i].set_xticks(range(10))
        axes[i].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Non-IID Label Distribution per Client")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def render_summary(
    fl_history: Dict[str, List[float]],
    dp_history: Optional[Dict[str, List[float]]] = None,
) -> str:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console(record=True, width=90)

    table = Table(title="PrivacyGuard FL — Training Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Standard FL", style="green")
    table.add_column("DP-FL", style="yellow")

    fl_acc = fl_history["test_accuracy"][-1] * 100
    fl_loss = fl_history["test_loss"][-1]

    if dp_history:
        dp_acc = dp_history["test_accuracy"][-1] * 100
        dp_loss = dp_history["test_loss"][-1]
        dp_eps = dp_history["privacy_spent"][-1]
        table.add_row("Final Accuracy", f"{fl_acc:.1f}%", f"{dp_acc:.1f}%")
        table.add_row("Final Loss", f"{fl_loss:.4f}", f"{dp_loss:.4f}")
        table.add_row("Privacy Budget (ε)", "None", f"{dp_eps:.2f}")
    else:
        table.add_row("Final Accuracy", f"{fl_acc:.1f}%", "—")
        table.add_row("Final Loss", f"{fl_loss:.4f}", "—")
        table.add_row("Privacy Budget (ε)", "None", "—")

    console.print(table)
    return console.export_text()
