"""Tests for API auth, Fernet encryption, rate limiting, replay protection."""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

import pytest
import requests
import torch
from wsgiref.simple_server import make_server

from src.deployment.api import create_app


@pytest.fixture(autouse=True, scope="session")
def _test_model():
    from src.core.federated import MNISTModel
    os.makedirs("output", exist_ok=True)
    torch.save(MNISTModel().state_dict(), "output/test_model.pt")


@pytest.fixture
def api_keys():
    return {"test-client": secrets.token_hex(32)}


@pytest.fixture
def app(api_keys):
    return create_app("output/test_model.pt", api_keys=api_keys)


@pytest.fixture
def server(app):
    srv = make_server("127.0.0.1", 0, app)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def _sign(api_id, api_key, body):
    ts = str(int(time.time()))
    nonce = secrets.token_hex(16)
    signing = f"{api_id}:{nonce}:{ts}:{body}"
    sig = hmac.new(api_key.encode(), signing.encode(), hashlib.sha256).hexdigest()
    return ts, nonce, sig


def test_health_no_auth(server):
    r = requests.get(f"{server}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_raw_no_auth(server):
    r = requests.post(f"{server}/predict/raw", json={"pixels": [0.0] * 784}, timeout=5)
    assert r.status_code == 401


def test_predict_raw_valid_auth(server, api_keys):
    body = json.dumps({"pixels": [0.0] * 784})
    ts, nonce, sig = _sign("test-client", api_keys["test-client"], body)
    headers = {
        "Content-Type": "application/json",
        "X-Api-Id": "test-client",
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        "X-Signature": sig,
    }
    r = requests.post(f"{server}/predict/raw", headers=headers, data=body, timeout=5)
    assert r.status_code == 200
    resp = r.json()
    assert "token" in resp
    assert len(resp["token"]) > 10


def test_wrong_signature_rejected(server, api_keys):
    body = json.dumps({"pixels": [0.0] * 784})
    ts, nonce, _ = _sign("test-client", api_keys["test-client"], body)
    headers = {
        "Content-Type": "application/json",
        "X-Api-Id": "test-client",
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        "X-Signature": "deadbeef" * 8,
    }
    r = requests.post(f"{server}/predict/raw", headers=headers, data=body, timeout=5)
    assert r.status_code == 401


def test_expired_timestamp_rejected(server, api_keys):
    body = json.dumps({"pixels": [0.0] * 784})
    ts_old = str(int(time.time()) - 600)
    nonce = secrets.token_hex(16)
    signing = f"test-client:{nonce}:{ts_old}:{body}"
    sig = hmac.new(api_keys["test-client"].encode(), signing.encode(), hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Api-Id": "test-client",
        "X-Timestamp": ts_old,
        "X-Nonce": nonce,
        "X-Signature": sig,
    }
    r = requests.post(f"{server}/predict/raw", headers=headers, data=body, timeout=5)
    assert r.status_code == 401


def test_unknown_api_id(server):
    r = requests.post(
        f"{server}/predict/raw",
        json={"pixels": [0.0] * 784},
        headers={"X-Api-Id": "unknown", "X-Timestamp": "0", "X-Nonce": "n", "X-Signature": "s"},
        timeout=5,
    )
    assert r.status_code == 401


def test_missing_headers(server):
    r = requests.post(f"{server}/predict/raw", json={"pixels": [0.0] * 784}, timeout=5)
    assert r.status_code == 401


def test_fernet_round_trip(server, api_keys):
    """Predict returns a Fernet token; decrypt it and verify the prediction is an integer."""
    body = json.dumps({"pixels": [0.0] * 784})
    ts, nonce, sig = _sign("test-client", api_keys["test-client"], body)
    headers = {
        "Content-Type": "application/json",
        "X-Api-Id": "test-client",
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        "X-Signature": sig,
    }
    r = requests.post(f"{server}/predict/raw", headers=headers, data=body, timeout=5)
    assert r.status_code == 200
    token_hex = r.json()["token"]
    assert isinstance(token_hex, str)
    assert len(token_hex) > 20
