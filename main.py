#!/usr/bin/env python3
"""PrivacyGuard FL — Privacy-Preserving Federated Learning with Attack Demo & Secure Deployment."""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

from src.core.federated import Client, FedServer, MNISTModel
from src.data.pipeline import DataPipe
from src.differential_privacy.dp_client import DPClient, DPFedServer
from src.attacks.gradient_inversion import GradientInversionAttack
from src.ui.visualization import (
    plot_attack_comparison,
    plot_label_distribution,
    plot_training_history,
    render_summary,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_fl_training(args):
    print("\n[1/4] Loading MNIST data with non-IID splits...")
    pipe = DataPipe(
        num_clients=args.clients,
        samples_per_client=args.samples,
        non_iid_degree=args.non_iid,
        batch_size=args.batch_size,
    )
    client_loaders, test_loader, client_indices = pipe.build_client_loaders()

    train_set, _ = pipe.load_full_dataset()
    dists = pipe.compute_label_distribution(train_set, client_indices)
    plot_label_distribution(dists, save_path="output/label_distribution.png")
    print(f"  -> {args.clients} clients, {args.samples} samples each (non-IID degree: {args.non_iid})")
    print("  -> Label distribution plot saved to output/label_distribution.png")

    device = get_device()
    print(f"\n[2/4] Training standard Federated Learning ({args.rounds} rounds)...")

    base_model = MNISTModel().to(device)
    clients = [
        Client(i, MNISTModel(), loader, local_epochs=args.local_epochs, device=device)
        for i, loader in enumerate(client_loaders)
    ]
    server = FedServer(base_model, clients, test_loader, rounds=args.rounds, device=device)

    fl_history = server.fit()

    fl_acc = fl_history["test_accuracy"][-1] * 100
    print(f"  -> Standard FL: Final accuracy = {fl_acc:.1f}%")

    torch.save(server.global_model.state_dict(), "output/fl_model.pt")

    if args.no_dp:
        plot_training_history(fl_history, save_path="output/training_history.png")
        return fl_history, None

    print(f"\n[3/4] Training DP-Federated Learning (ε={args.epsilon})...")
    dp_clients = [
        DPClient(
            i,
            MNISTModel(),
            loader,
            epsilon=args.epsilon,
            learning_rate=args.lr_dp,
            local_epochs=args.local_epochs,
            total_rounds=args.rounds,
            device=device,
        )
        for i, loader in enumerate(client_loaders)
    ]
    dp_server = DPFedServer(MNISTModel(), dp_clients, test_loader, rounds=args.rounds, device=device)
    dp_history = dp_server.fit()

    dp_acc = dp_history["test_accuracy"][-1] * 100
    dp_eps = dp_history["privacy_spent"][-1]
    print(f"  -> DP-FL: Final accuracy = {dp_acc:.1f}% (Privacy budget spent: ε={dp_eps:.2f})")

    torch.save(dp_server.global_model.state_dict(), "output/dp_model.pt")
    torch.save(dp_server.global_model.state_dict(), "models/final_model.pt")

    plot_training_history(fl_history, dp_history, save_path="output/training_history.png")
    summary = render_summary(fl_history, dp_history)
    print(f"\n{summary}")
    print("  -> Training history plot saved to output/training_history.png")

    return fl_history, dp_history


def run_attack_demo(args):
    print("\n[Attack Demo] Gradient Inversion Attack (iDLG + Adam)")
    print("  Capturing a single-sample gradient from client 0's fresh model")
    print("  and reconstructing the private image via iDLG.\n")

    device = get_device()
    pipe = DataPipe(
        num_clients=args.clients,
        samples_per_client=args.samples,
        non_iid_degree=args.non_iid,
        batch_size=1,
    )
    client_loaders, _, _ = pipe.build_client_loaders()

    data, target = next(iter(client_loaders[0]))
    data, target = data.to(device), target.to(device)

    model = MNISTModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # --- Unprotected gradient ---
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    original_weights = {n: p.cpu().numpy().copy() for n, p in model.state_dict().items()}
    unprotected_grad = {n: p.grad.cpu().numpy().copy() for n, p in model.named_parameters() if p.grad is not None}

    # --- DP-protected gradient (same input, same weights, through DP pipeline) ---
    dp_client = DPClient(
        0, MNISTModel(), client_loaders[0],
        epsilon=args.epsilon,
        local_epochs=1,
        total_rounds=1,
        device=device,
    )
    dp_client.model.load_state_dict({n: torch.from_numpy(v).to(device) for n, v in original_weights.items()})
    dp_grad = dp_client.compute_dp_gradient(data, target)

    print(f"  DP params: σ={dp_client.noise_multiplier:.4f}  C={dp_client.clip_norm}  "
          f"q={dp_client.sample_rate:.4f}  ε={args.epsilon}")
    print(f"  True label: {int(target[0].item())}")
    print()

    # --- Attack unprotected gradient ---
    print("  == Attack on unprotected (No-DP) gradient ==")
    attack_nd = GradientInversionAttack(model, original_weights, unprotected_grad, device=device)
    recon_no_dp, inferred_label, _ = attack_nd.attack(
        batch_size=1,
        steps=args.attack_steps,
        learning_rate=args.attack_lr,
        label_hint=None,
        verbose=True,
    )
    mse_nd = GradientInversionAttack.evaluate_reconstruction_mse(data, recon_no_dp)
    ssim_nd = GradientInversionAttack.evaluate_ssim(data, recon_no_dp)
    print(f"    => iDLG inferred label: {inferred_label}  |  "
          f"MSE = {mse_nd:.6f}   SSIM = {ssim_nd:.4f}\n")

    # --- Attack DP-protected gradient ---
    print("  == Attack on DP-protected gradient ==")
    attack_dp = GradientInversionAttack(model, original_weights, dp_grad, device=device)
    recon_dp, inferred_label_dp, _ = attack_dp.attack(
        batch_size=1,
        steps=args.attack_steps,
        learning_rate=args.attack_lr,
        label_hint=None,
        verbose=True,
    )
    mse_dp = GradientInversionAttack.evaluate_reconstruction_mse(data, recon_dp)
    ssim_dp = GradientInversionAttack.evaluate_ssim(data, recon_dp)
    print(f"    => iDLG inferred label: {inferred_label_dp}  |  "
          f"MSE = {mse_dp:.6f}   SSIM = {ssim_dp:.4f}\n")

    # --- Summary ---
    print("  " + "─" * 60)
    print(f"  {"Attack Efficacy Summary":^60}")
    print("  " + "─" * 60)
    print(f"  {"":>24} {"MSE":>12} {"SSIM":>12}")
    print(f"  {"No DP (unprotected)":>24} {mse_nd:>12.6f} {ssim_nd:>12.4f}")
    print(f"  {"With DP (protected)":>24} {mse_dp:>12.6f} {ssim_dp:>12.4f}")
    mse_ratio = mse_dp / (mse_nd + 1e-8)
    print(f"  {"DP / No-DP ratio":>24} {mse_ratio:>17.1f}x")
    print("  " + "─" * 60)

    plot_attack_comparison(data, recon_no_dp, recon_dp, save_path="output/attack_comparison.png")
    print("\n  => 3-panel figure saved to output/attack_comparison.png")

    return {"mse_no_dp": mse_nd, "ssim_no_dp": ssim_nd, "mse_dp": mse_dp, "ssim_dp": ssim_dp}


def run_start_server(args):
    from src.deployment.api import run_server
    model_path = args.model_path or "output/dp_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Train first with: python main.py train")
        sys.exit(1)
    print(f"\n[Deploy] Starting authenticated prediction API on {args.host}:{args.port}...")
    run_server(
        model_path,
        host=args.host,
        port=args.port,
        api_id=args.api_id if getattr(args, "api_id", None) else None,
        api_key=args.api_key if getattr(args, "api_key", None) else None,
    )


def main():
    parser = argparse.ArgumentParser(
        description="PrivacyGuard FL — Privacy-Preserving Federated Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                          # Run full training + DP
  python main.py train --no-dp                  # Standard FL only
  python main.py attack                         # Gradient inversion demo
  python main.py deploy                         # Start secure API server
  python main.py full                           # Train, attack, deploy
        """,
    )

    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Train Federated Learning models")
    train.add_argument("--clients", type=int, default=4)
    train.add_argument("--samples", type=int, default=1500)
    train.add_argument("--non-iid", type=float, default=0.8)
    train.add_argument("--rounds", type=int, default=20)
    train.add_argument("--local-epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--epsilon", type=float, default=8.0)
    train.add_argument("--lr-dp", type=float, default=0.05)
    train.add_argument("--no-dp", action="store_true")

    attack = sub.add_parser("attack", help="Run Gradient Inversion attack demo")
    attack.add_argument("--clients", type=int, default=4)
    attack.add_argument("--samples", type=int, default=1500)
    attack.add_argument("--non-iid", type=float, default=0.8)
    attack.add_argument("--epsilon", type=float, default=8.0, help="DP epsilon for protected gradient")
    attack.add_argument("--attack-steps", type=int, default=1000, help="Adam optimization steps")
    attack.add_argument("--attack-lr", type=float, default=0.1, help="Adam learning rate")

    deploy = sub.add_parser("deploy", help="Start authenticated prediction API")
    deploy.add_argument("--host", default="0.0.0.0")
    deploy.add_argument("--port", type=int, default=5000)
    deploy.add_argument("--model-path", default="output/dp_model.pt")
    deploy.add_argument("--api-id", default=None, help="API client ID (auto-generated if omitted)")
    deploy.add_argument("--api-key", default=None, help="API secret key (auto-generated if omitted)")

    full = sub.add_parser("full", help="Train, attack, and start server")
    full.add_argument("--clients", type=int, default=4)
    full.add_argument("--samples", type=int, default=1500)
    full.add_argument("--rounds", type=int, default=15)
    full.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if args.command == "train":
        run_fl_training(args)

    elif args.command == "attack":
        run_attack_demo(args)

    elif args.command == "deploy":
        run_start_server(args)

    elif args.command == "full":
        print("=" * 60)
        print("PrivacyGuard FL — Full Pipeline".center(60))
        print("=" * 60)
        run_fl_training(args)
        run_attack_demo(args)

        print("\n" + "=" * 60)
        print("Training complete. Start the API with:")
        print(f"  python main.py deploy --port {args.port}")
        print("=" * 60)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
