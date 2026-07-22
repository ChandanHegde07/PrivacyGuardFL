# PrivacyGuard FL

Privacy-Preserving Federated Learning with Attack Demonstration & Secure Deployment.

Federated Learning system that trains MNIST collaboratively across 4 simulated
clients (hospitals) without sharing raw data, adds Differential Privacy
(DP-SGD with Opacus-verified RDP accounting), demonstrates Gradient Inversion
attacks, and deploys the DP-protected model behind a Fernet-encrypted Flask API.

## Architecture

```
                       Non-IID data split ──► Client 0 (digits 0–1)
                      ╱                      Client 1 (digits 1–2)
  MNIST ── DataPipe ──┼── 500–1500 each ──── Client 2 (digits 2–3)
                      ╲                      Client 3 (digits 3–4)
                       local epochs × 5 rounds

                          │  model weights  │
                          ▼                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                     FedAvg Aggregation                       │
  │                                                              │
  │  Standard FL path:    Σ(weights) / N     (no protection)     │
  │        DP-FL path:    Σ(weights) / N     on DP-SGD updates   │
  │                       ↑ per-sample clip C=1.0                  │
  │                       ↑ Gaussian noise σ·C in each step       │
  │                       ↑ ε tracked via Opacus RDP accountant   │
  └──────────────┬───────────────────────────┬───────────────────┘
                 │                           │
    ┌────────────▼───────────┐   ┌───────────▼──────────────────┐
    │  Gradient Inversion    │   │  Secure Flask API             │
    │  Attack Demo           │   │                               │
    │                        │   │  POST /predict   (image file) │
    │  DLG: optimise dummy   │   │  POST /predict/raw (28×28)   │
    │  inputs to match       │   │  POST /health                 │
    │  leaked gradients      │   │                               │
    │                        │   │  Fernet (AES-128-CBC)         │
    │  Shows data leakage    │   │  encrypts predictions         │
    │  without DP            │   │                               │
    └────────────────────────┘   └──────────────────────────────┘
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Train both FL and DP-FL (default: 4 clients, 1500 samp, 20 rounds, ε=8)
python main.py train

# Gradient Inversion attack demo
python main.py attack

# Deploy DP model behind encrypted API
python main.py deploy --port 5000

# Full pipeline
python main.py full
```

## Commands

### `train` — Federated training

| Flag | Default | Description |
|------|---------|-------------|
| `--clients` | 4 | Number of simulated clients |
| `--samples` | 1500 | Training samples per client |
| `--non-iid` | 0.8 | Label-skew degree (0 = uniform, 1 = extreme) |
| `--rounds` | 20 | Federated aggregation rounds |
| `--local-epochs` | 3 | SGD epochs per client per round |
| `--batch-size` | 32 | Local batch size |
| `--epsilon` | 8.0 | DP privacy budget ε (δ fixed at 1e-5) |
| `--lr-dp` | 0.05 | Learning rate for DP clients |
| `--no-dp` | — | Skip DP training, run standard FL only |

```bash
python main.py train                              # default ε=8, 20 rounds
python main.py train --no-dp                      # standard FL only
python main.py train --epsilon 4.0 --rounds 30    # tight privacy budget
python main.py train --samples 3000 --clients 6   # more data, more clients
```

### `attack` — Gradient Inversion demo

| Flag | Default | Description |
|------|---------|-------------|
| `--clients` | 4 | Clients for data splitting |
| `--samples` | 1500 | Samples per client |
| `--non-iid` | 0.8 | Label-skew degree |
| `--attack-steps` | 300 | DLG optimization steps |
| `--attack-lr` | 0.1 | Attack learning rate |

```bash
python main.py attack
python main.py attack --attack-steps 500 --attack-lr 0.05
```

### `deploy` — Secure prediction API

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 5000 | Listen port |
| `--model-path` | output/dp_model.pt | Trained model checkpoint |

```bash
python main.py deploy --port 8080
curl -X POST http://localhost:8080/predict/raw \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0.0, 0.0, ...]}'   # 784 float values
```

### `full` — Train, attack, print deploy instructions

```bash
python main.py full --rounds 15 --samples 1500
```

## Output

All artifacts land in `output/`:

| File | Produced by |
|------|-------------|
| `label_distribution.png` | `train` — per-client digit distribution |
| `training_history.png` | `train` — accuracy, loss, ε over rounds |
| `fl_model.pt` | `train` — standard FL checkpoint |
| `dp_model.pt` | `train` — DP-FL checkpoint |
| `attack_comparison.png` | `attack` — original vs reconstructed image |

## Modules

```
src/
├── data/pipeline.py              Non-IID MNIST → per-client DataLoaders
├── core/federated.py             MNISTModel, Client, FedServer, FedAvg
├── differential_privacy/
│   └── dp_client.py              DPClient (clip + noise + Opacus RDP),
│                                 DPFedServer, ε tracking per round
├── attacks/
│   └── gradient_inversion.py     DLG attack: reconstruct inputs from leaked grads
├── deployment/
│   └── api.py                    Flask app: /predict, /predict/raw, /health
│                                 Fernet-encrypted responses
└── ui/
    └── visualization.py          Matplotlib plots + Rich terminal summary
```

## Privacy Guarantees

- **DP-SGD**: Each client clips per-sample gradients to `C=1.0`,
  adds Gaussian noise `N(0, σ²C²)`, and tracks (ε,δ)-DP via Opacus RDP
  composition. `σ` is computed from `get_noise_multiplier` for the
  target ε,δ, sample rate q, and total epochs.

- **Verified**: Round-by-round ε matches Opacus `RDPAccountant` exactly
  (verified during audit; see commit history).

- **Attack demonstration**: The Gradient Inversion (DLG) attack shows
  that without DP, an adversary intercepting model updates can reconstruct
  recognizable training images. With DP (σ > 0), reconstruction quality
  degrades proportionally.

- **Deployment**: Predictions are encrypted with Fernet
  (AES-128-CBC + HMAC-SHA256). The encryption key is printed at server
  startup — save it to decrypt responses.
