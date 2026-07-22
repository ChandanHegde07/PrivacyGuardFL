# PrivacyGuard FL

Privacy-Preserving Federated Learning with Attack Demonstration & Secure Deployment.

A Federated Learning system that trains a model collaboratively without sharing raw data,
adds Differential Privacy for protection, demonstrates real privacy attacks (Gradient Inversion),
and deploys the model as a secure web API.

## Architecture

```
                    ┌──────────────┐
                    │   Clients     │
                    │ (Hospitals)   │
                    │              │
                    │ Train locally │──── shared model updates ────┐
                    │ on own data   │                              │
                    └──────────────┘                              │
                                                                ▼
    Gradient Inversion ◄──── Captured gradients    ┌─────────────────────┐
    Attack (demonstrates                          │  FL / DP-FL Server   │
    data leakage risk)                            │                     │
                                                  │ FedAvg aggregation   │
                                                  │ DP noise addition    │
                                                  └─────────┬───────────┘
                                                            │
                                                            ▼
                                                  ┌─────────────────────┐
                                                  │  Secure API (Flask)  │
                                                  │  Encrypted responses │
                                                  └─────────────────────┘
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train Models

```bash
# Full training: Standard FL + Differential Privacy FL
python main.py train

# Fast test run with fewer rounds
python main.py train --rounds 10 --samples 500

# Standard FL only (no DP)
python main.py train --no-dp

# Custom DP parameters
python main.py train --epsilon 4.0 --rounds 30 --clients 6
```

### Run Privacy Attack Demo

```bash
# Demonstrate Gradient Inversion (DLG) attack
python main.py attack

# With custom attack parameters
python main.py attack --attack-steps 500 --attack-lr 0.05
```

### Deploy Secure API

```bash
# Start prediction server with encryption
python main.py deploy --port 5000

# Test the API
curl -X POST http://localhost:5000/predict/raw \
  -H "Content-Type: application/json" \
  -d '{"pixels": [...]}'    # 784 pixel values
```

### Full Pipeline

```bash
python main.py full    # Train → Attack → Ready to deploy
```

## CLI Options

| Command | Option | Default | Description |
|---------|--------|---------|-------------|
| `train` | `--clients` | 4 | Number of simulated clients |
| `train` | `--samples` | 1500 | Samples per client |
| `train` | `--non-iid` | 0.8 | Non-IID degree (0=uniform, 1=extreme skew) |
| `train` | `--rounds` | 20 | Federated rounds |
| `train` | `--local-epochs` | 3 | Local epochs per round |
| `train` | `--epsilon` | 8.0 | DP privacy budget |
| `train` | `--no-dp` | — | Skip DP training |
| `attack` | `--attack-steps` | 300 | Gradient inversion optimization steps |
| `attack` | `--attack-lr` | 0.1 | Attack learning rate |
| `deploy` | `--host` | 0.0.0.0 | API bind address |
| `deploy` | `--port` | 5000 | API port |
| `deploy` | `--model-path` | output/dp_model.pt | Trained model file |

## Output

All results are saved to the `output/` directory:

- `output/label_distribution.png` — Non-IID data split visualization
- `output/training_history.png` — Accuracy/loss/privacy budget over rounds
- `output/attack_comparison.png` — Original vs. reconstructed images from the attack
- `output/fl_model.pt` — Standard FL trained model
- `output/dp_model.pt` — DP-protected trained model

## Modules

| Module | Path | Description |
|--------|------|-------------|
| Data Pipeline | `src/data/pipeline.py` | Non-IID MNIST splitting across clients |
| FL Core | `src/core/federated.py` | Client, Server, FedAvg, standard training |
| Differential Privacy | `src/differential_privacy/dp_client.py` | DP-SGD client, DP server with ε tracking |
| Gradient Inversion | `src/attacks/gradient_inversion.py` | DLG attack reconstructing inputs from gradients |
| Deployment | `src/deployment/api.py` | Flask API with Fernet-encrypted responses |
| Visualization | `src/ui/visualization.py` | Matplotlib plots, Rich terminal summaries |

## Privacy Details

- **Differential Privacy**: Per-sample gradient clipping + calibrated Gaussian noise
  added during local training. Each client's privacy budget (ε) is tracked across rounds.
- **Gradient Inversion Attack**: Demonstrates that without DP, an adversary who
  intercepts model updates can reconstruct training images via DLG optimization.
- **Secure Deployment**: Model predictions are encrypted with Fernet (AES-128-CBC +
  HMAC-SHA256) before transmission.
