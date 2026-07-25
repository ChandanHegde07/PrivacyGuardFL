# PrivacyGuard FL

Privacy-preserving federated learning on MNIST with DP-SGD, gradient-inversion attack demo, and authenticated API deployment.

```
MNIST ──► Non-IID split ──► 4 clients ──► FedAvg ──┬─► Standard FL
                                                    └─► DP-FL (clip C + noise σ·C/bs + Opacus ε)
                                                         │
                      Gradient Inversion Attack ◄────────┘
                                                         │
                      Authenticated Flask API ◄──────────┘
                        (HMAC auth + Fernet + rate limit)
```

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python main.py full                  # train → attack → summary
```

## Commands

| Command | What it does |
|---------|--------------|
| `python main.py train` | Trains Standard FL + DP-FL (ε=8). Saves models + plots. |
| `python main.py attack` | iDLG reconstruction from unprotected vs DP-protected gradients. |
| `python main.py deploy` | Starts authenticated prediction API on `:5000`. |
| `python main.py full` | train → attack → print deploy instructions. |

### `train` options

| Flag | Default | Description |
|------|---------|-------------|
| `--clients` | 4 | Simulated hospitals |
| `--samples` | 1500 | Training samples per client |
| `--non-iid` | 0.8 | Label skew (0=uniform, 1=extreme) |
| `--rounds` | 20 | FL aggregation rounds |
| `--local-epochs` | 3 | SGD epochs per round per client |
| `--batch-size` | 16 | Local batch size |
| `--epsilon` | 8.0 | DP privacy budget (δ=1e-5) |
| `--lr-dp` | 0.3 | DP learning rate (no momentum) |
| `--no-dp` | — | Skip DP-FL, only standard FL |

### `attack` options

| Flag | Default | Description |
|------|---------|-------------|
| `--attack-steps` | 1000 | Adam iterations |
| `--attack-lr` | 0.1 | Adam learning rate |

### `deploy` options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 5000 | Listen port |
| `--model-path` | output/dp_model.pt | Model checkpoint |
| `--api-id` | auto | Client ID |
| `--api-key` | auto | HMAC secret |

### Deploy with auth

```bash
python main.py deploy --port 5000

API_ID="<from-stdout>" API_KEY="<from-stdout>"
BODY='{"pixels": [0.0, ... 784 values ...]}'
TS=$(date +%s) NONCE=$(openssl rand -hex 16)
SIG=$(echo -n "${API_ID}:${NONCE}:${TS}:${BODY}" | openssl dgst -sha256 -hmac "$API_KEY" | cut -d' ' -f2)
curl -X POST http://localhost:5000/predict/raw \
  -H "Content-Type: application/json" \
  -H "X-Api-Id: ${API_ID}" -H "X-Timestamp: ${TS}" \
  -H "X-Nonce: ${NONCE}" -H "X-Signature: ${SIG}" \
  -d "$BODY"
```

### Docker

```bash
docker compose build
docker compose run train
docker compose run attack
docker compose up api
```

## Results (seed 42, 20 rounds, 1500 samples/client)

| Metric | No-DP | DP (ε=8) |
|--------|-------|----------|
| Test accuracy | 97.2% | 84.0% |
| Attack MSE | 0.72 | 1.30 |
| Attack SSIM | 0.12 | 0.14 |

## DP-SGD Parameters

| Param | Value | Why |
|-------|-------|-----|
| Clip norm C | 1.0 | Per-sample gradient bound |
| ε | 8.0 | Privacy budget (configurable) |
| δ | 1e-5 | Failure probability |
| σ | computed via Opacus | Noise multiplier from ε, δ, q, epochs |
| Noise on averaged grad | `σ·C / batch_size` | PyTorch `.backward()` returns mean, not sum |
| Learning rate | 0.3 | Higher LR compensates for noise (DP has no momentum) |
| Local epochs | 3 | More steps per round averages noise |

**Noise bugfix** (`dp_client.py:105`): noise std was `σ·C` — `batch_size`× too large for averaged gradients. Divided by `batch_size` to match DP-SGD accounting.

## API Security

| Property | Mechanism |
|----------|-----------|
| Auth | HMAC-SHA256 per-request (X-Signature) |
| Confidentiality | Fernet (AES-128-CBC + HMAC-SHA256) |
| Replay protection | Timestamp ±300s + per-request nonce |
| Rate limiting | 120 req / 60s per IP |
| Transport | Not provided — use a reverse proxy for TLS |

## Output

| File | Content |
|------|---------|
| `output/training_curves.png` | Accuracy/loss over rounds |
| `output/privacy_budget.png` | ε spent per round |
| `output/attack_comparison.png` | Original / no-DP recon / DP recon |
| `output/label_distribution.png` | Non-IID skew per client |
| `output/dp_model.pt` | Trained model checkpoint |
| `models/final_model.pt` | Canonical model location |

## Project Layout

```
PrivacyGuard FL/
├── main.py, pyproject.toml, setup.py, requirements.txt
├── Dockerfile, docker-compose.yml
├── .github/workflows/ci.yml
├── src/
│   ├── data/pipeline.py
│   ├── core/federated.py
│   ├── differential_privacy/dp_client.py
│   ├── attacks/gradient_inversion.py
│   ├── deployment/api.py
│   └── ui/visualization.py
├── tests/ (5 test files, 25 tests)
└── notebooks/01_end_to_end_demo.ipynb
```

## Limitations

- Simulated clients only (single process, sequential local training)
- No TLS — wrap with nginx/Caddy for production
- MNIST only — CNN generalizes but data pipeline would need adaptation
- Notebook requires jupyter kernel with `.venv` activated

## Tests

```bash
pytest tests/ -v -m "not slow"
```
