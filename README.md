# PrivacyGuard FL

Privacy-Preserving Federated Learning with Attack Demonstration & Authenticated API Deployment.

Federated Learning system that trains MNIST collaboratively across 4 simulated clients (hospitals)
without sharing raw data, adds Differential Privacy (DP-SGD with Opacus RDP accounting),
demonstrates Gradient Inversion (iDLG) attacks, and deploys the DP-protected model behind
an authenticated, rate-limited, Fernet-encrypted Flask API.

## Architecture

```
                    Non-IID data split ──► Client 0 (digits 0–1)
                   ╱                      Client 1 (digits 1–2)
  MNIST ── DataPipe ─┼── 500–1500 each ──── Client 2 (digits 2–3)
                   ╲                      Client 3 (digits 3–4)
                    local epochs × N rounds

                       │  model weights  │
                       ▼                 ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                     FedAvg Aggregation                        │
  │                                                               │
  │  Standard FL:     Σ(weights) / N     (no protection)          │
  │                                                               │
  │  DP-FL:           Σ(weights) / N     on DP-SGD updates        │
  │                           ↑ per-sample clip C=1.0              │
  │                           ↑ Gaussian noise σ·C per step        │
  │                           ↑ ε via Opacus RDP accountant        │
  └──────────────┬────────────────────────────┬───────────────────┘
                 │                            │
    ┌────────────▼───────────┐   ┌────────────▼──────────────────┐
    │  Gradient Inversion    │   │  Authenticated Flask API      │
    │  Attack Demo           │   │                                │
    │                        │   │  POST /predict    (image)      │
    │  iDLG: infer label     │   │  POST /predict/raw (28×28)     │
    │  from sign(∇b_fc2)     │   │  GET  /health                  │
    │                        │   │                                │
    │  Adam gradient         │   │  HMAC-SHA256 per-request auth  │
    │  matching + TV reg     │   │  IP rate limiting              │
    │                        │   │  Fernet encrypted responses    │
    │  Cosine similarity     │   │  Timestamp+nonce replay guard  │
    │  loss on all layers    │   │                                │
    └────────────────────────┘   └───────────────────────────────┘
```

## Quick Start

```bash
# 1. Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Run everything (train → attack → summary)
python main.py full
```

## Setup

| Method | Command |
|--------|---------|
| pip (editable) | `pip install -e .` |
| pip (deps)     | `pip install -r requirements.txt` |
| Docker         | `docker compose build && docker compose run train` |

## Commands

### `python main.py train`

Trains both Standard FL and DP-FL, saves models and plots.

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
python main.py train                              # default 20 rounds, ε=8
python main.py train --no-dp                      # standard FL only
python main.py train --epsilon 4.0 --rounds 30    # tight privacy
python main.py train --samples 3000 --clients 6   # more data
```

### `python main.py attack`

Runs iDLG gradient inversion on unprotected and DP-protected gradients from a single input,
reports MSE and SSIM for both, and saves a 3-panel comparison figure.

| Flag | Default | Description |
|------|---------|-------------|
| `--clients` | 4 | Clients for data splitting |
| `--samples` | 1500 | Samples per client |
| `--non-iid` | 0.8 | Label-skew degree |
| `--epsilon` | 8.0 | DP epsilon for the protected gradient |
| `--attack-steps` | 1000 | Adam optimization steps |
| `--attack-lr` | 0.1 | Adam learning rate |

### `python main.py deploy`

Starts the authenticated prediction API.

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 5000 | Listen port |
| `--model-path` | output/dp_model.pt | Trained model |
| `--api-id` | auto | Client ID |
| `--api-key` | auto | HMAC signing secret |

```bash
python main.py deploy --port 5000

# Client call
API_ID="<printed-at-startup>"
API_KEY="<printed-at-startup>"
BODY='{"pixels": [0.0, ... 784 values ...]}'
TIMESTAMP=$(date +%s)
NONCE=$(openssl rand -hex 16)
SIG=$(echo -n "${API_ID}:${NONCE}:${TIMESTAMP}:${BODY}" | openssl dgst -sha256 -hmac "$API_KEY" | cut -d' ' -f2)
curl -X POST http://localhost:5000/predict/raw \
  -H "Content-Type: application/json" \
  -H "X-Api-Id: ${API_ID}" \
  -H "X-Timestamp: ${TIMESTAMP}" \
  -H "X-Nonce: ${NONCE}" \
  -H "X-Signature: ${SIG}" \
  -d "$BODY"
```

### `python main.py full`

Runs train, then attack, then prints deploy instructions.

### Docker

```bash
# Build
docker compose build

# Train
docker compose run train

# Attack
docker compose run attack

# API server
docker compose up api
```

## Output

### Plots (generated by `train`, `attack`, or `full`)

| File | Description |
|------|-------------|
| `output/label_distribution.png` | Non-IID label skew across 4 clients |
| `output/training_curves.png` | Accuracy and loss over rounds (FL vs DP-FL) |
| `output/privacy_budget.png` | ε spent per round vs target budget |
| `output/attack_comparison.png` | 3-panel: original \| no-DP recon \| DP recon with MSE/SSIM |

### Model checkpoints

| File | Description |
|------|-------------|
| `output/fl_model.pt` | Standard FL trained model |
| `output/dp_model.pt` | DP-FL trained model |
| `models/final_model.pt` | Same as dp_model.pt (canonical location) |

## Modules

```
src/
├── __init__.py
├── data/pipeline.py                  Non-IID MNIST → per-client DataLoaders
├── core/federated.py                 MNISTModel, Client, FedServer, FedAvg
├── differential_privacy/
│   └── dp_client.py                  DPClient, DPFedServer, Opacus RDP ε tracking
├── attacks/
│   └── gradient_inversion.py         iDLG gradient inversion (Adam + TV reg)
├── deployment/
│   └── api.py                        Flask API: HMAC auth, Fernet, rate limiting
└── ui/
    └── visualization.py              Matplotlib plots + Rich terminal tables

notebooks/
└── 01_end_to_end_demo.ipynb          Walkthrough with markdown explanations

tests/
├── test_data.py                      Non-IID split, shapes, label skew
├── test_federated.py                 FedAvg, Client, Server
├── test_dp.py                        Clipping, noise, ε increase
├── test_attack.py                    SSIM, label inference, attack shapes
└── test_api.py                       HMAC auth, Fernet, rate limiting, replay
```

## Privacy Guarantees

### DP-SGD Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Clip norm `C` | 1.0 | Per-sample gradient L2 bound |
| Target ε | 8.0 | Privacy budget (configurable via `--epsilon`) |
| δ | 1e-5 | Failure probability |
| Noise multiplier σ | Computed from ε,δ,q,epochs | Gaussian noise `N(0, σ²C²)` |
| Accountant | Opacus RDP | Per-step moments, composed via RDP |

The noise multiplier σ is computed via Opacus `get_noise_multiplier()`
and verified against Opacus `RDPAccountant` at every training round.

### API Security Properties

| Property | Mechanism | Notes |
|----------|-----------|-------|
| Authentication | HMAC-SHA256 per-request signing | API ID + shared secret |
| Payload confidentiality | Fernet (AES-128-CBC + HMAC-SHA256) | Response body must be decrypted client-side |
| Rate limiting | Per-IP sliding window (120 req / 60 s) | Prevents model extraction |
| Input validation | Max image size 10 MB, max payload 64 KB, pixel count check | Blocks malformed requests |
| Transport security | Not provided | Use a reverse proxy with TLS for production |
| Replay protection | Timestamp ±300 s window + per-request nonce | Stale requests are rejected |

## Limitations

- **Simulated clients only**: All clients run locally; no real network communication.
  The "Federated Learning" is simulated via sequential local training + FedAvg aggregation.
- **No TLS**: The Flask API runs on plain HTTP. Wrap with nginx/Caddy for TLS.
- **Single task**: Currently MNIST digit classification only.
  The pipeline generalizes to any image model with minimal changes.
- **Small-scale DP**: With 1500 samples per client, DP-SGD noise is high relative to signal.
  Increasing `--samples` and `--local-epochs` improves DP accuracy.
- **Notebook dependencies**: The notebook requires a running Jupyter kernel with the .venv.
  Launch with `jupyter notebook` from the project root after activating the venv.

## Tests

```bash
pip install pytest
pytest tests/ -v -m "not slow"
```

## Repository

```
.gitignore
LICENSE
README.md
pyproject.toml
setup.py
requirements.txt
Dockerfile
docker-compose.yml
.github/workflows/ci.yml
main.py
data/          (downloaded MNIST — gitignored)
models/        (model checkpoints — gitignored)
output/        (generated plots — gitignored)
notebooks/
src/
tests/
```
