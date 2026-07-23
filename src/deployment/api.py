import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict
from functools import wraps
from io import BytesIO
from typing import Dict, Optional

import numpy as np
import torch
from cryptography.fernet import Fernet
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

from ..core.federated import MNISTModel

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PAYLOAD_SIZE = 64 * 1024
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 120
REQUEST_SIGNATURE_WINDOW = 300


def _authorized(api_keys: Dict[str, str]):
    """Require a valid HMAC signature in the X-Signature header.

    The client computes:
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        signature = base64(hmac-sha256(api_key, api_id:nonce:timestamp:body))

    and sends headers: X-Api-Id, X-Timestamp, X-Nonce, X-Signature.
    """

    def _json_body():
        data = request.get_data(as_text=True)
        if len(data) > MAX_PAYLOAD_SIZE:
            return None
        return data

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            api_id = request.headers.get("X-Api-Id")
            sig = request.headers.get("X-Signature")
            ts_str = request.headers.get("X-Timestamp")
            nonce = request.headers.get("X-Nonce")

            if not all([api_id, sig, ts_str, nonce]):
                return jsonify({"error": "Missing authentication headers"}), 401

            if api_id not in api_keys:
                return jsonify({"error": "Unknown API ID"}), 401

            try:
                timestamp = int(ts_str)
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid timestamp"}), 401

            if abs(int(time.time()) - timestamp) > REQUEST_SIGNATURE_WINDOW:
                return jsonify({"error": "Request expired or clock skew too large"}), 401

            body = _json_body()
            if body is None:
                return jsonify({"error": "Payload too large"}), 413

            signing_string = f"{api_id}:{nonce}:{timestamp}:{body}"
            expected = hmac.new(
                api_keys[api_id].encode(), signing_string.encode(), hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(expected, sig):
                return jsonify({"error": "Invalid signature"}), 401

            return f(*args, **kwargs)

        return wrapper

    return decorator


def create_app(
    model_path: str,
    api_keys: Dict[str, str],
    fernet_key: Optional[bytes] = None,
    rate_limit: bool = True,
) -> Flask:
    app = Flask(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if fernet_key is None:
        fernet_key = Fernet.generate_key()

    cipher = Fernet(fernet_key)
    app.config["FERNET_KEY"] = fernet_key

    if rate_limit:
        _rate_buckets = defaultdict(list)

        def _check_rate():
            now = time.time()
            ip = request.remote_addr or "unknown"
            bucket = _rate_buckets[ip]
            bucket[:] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]
            bucket.append(now)
            if len(bucket) > RATE_LIMIT_MAX_REQUESTS:
                return jsonify({"error": "Rate limit exceeded"}), 429
            return None
    else:
        def _check_rate():
            return None

    authorized = _authorized(api_keys)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model_loaded": True})

    @app.route("/predict", methods=["POST"])
    @authorized
    def predict():
        limit = _check_rate()
        if limit:
            return limit

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        header = file.read(MAX_IMAGE_BYTES + 1)
        if len(header) > MAX_IMAGE_BYTES:
            return jsonify({"error": "Image too large"}), 413

        try:
            img = Image.open(BytesIO(header))
        except Exception:
            return jsonify({"error": "Invalid image"}), 400

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = int(output.argmax(dim=1).item())

        payload = f"{prediction}".encode()
        token = cipher.encrypt(payload)

        return jsonify({"token": token.hex()})

    @app.route("/predict/raw", methods=["POST"])
    @authorized
    def predict_raw():
        limit = _check_rate()
        if limit:
            return limit

        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        raw_values = data.get("pixels")
        if not raw_values or len(raw_values) != 784:
            return jsonify({"error": "Expected 784 pixel values"}), 400

        try:
            arr = np.array(raw_values, dtype=np.float32)
        except (ValueError, TypeError):
            return jsonify({"error": "Pixels must be numeric"}), 400

        img = Image.fromarray((np.clip(arr.reshape(28, 28) * 255, 0, 255)).astype(np.uint8))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = int(output.argmax(dim=1).item())

        payload = f"{prediction}".encode()
        token = cipher.encrypt(payload)

        return jsonify({"token": token.hex()})

    return app


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 5000,
    api_id: Optional[str] = None,
    api_key: Optional[str] = None,
):
    if api_id is None:
        api_id = "client-" + secrets.token_hex(4)
    if api_key is None:
        api_key = secrets.token_hex(32)

    fernet_key = Fernet.generate_key()

    print(f"\n  API ID:       {api_id}")
    print(f"  API Key:      {api_key}")
    print(f"  Fernet Key:   {fernet_key.decode()}")
    print(f"\n  Save these credentials. Without them, no client can call this server.\n")

    app = create_app(model_path, api_keys={api_id: api_key}, fernet_key=fernet_key)
    app.run(host=host, port=port, debug=False, ssl_context=None)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/global_model.pt"
    run_server(model_path)
