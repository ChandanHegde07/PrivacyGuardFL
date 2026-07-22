import os
import tempfile
from io import BytesIO
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from cryptography.fernet import Fernet
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

from ..core.federated import MNISTModel


def create_app(model_path: str, encryption_key: Optional[bytes] = None) -> Flask:
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

    if encryption_key is None:
        encryption_key = Fernet.generate_key()

    cipher = Fernet(encryption_key)
    app.config["MODEL_KEY"] = encryption_key

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model_loaded": True})

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        img = Image.open(BytesIO(file.read()))

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            prediction = int(output.argmax(dim=1).item())

        encrypted_response = cipher.encrypt(str(prediction).encode())

        return jsonify({
            "prediction": prediction,
            "confidence": float(probabilities[prediction]),
            "probabilities": probabilities.tolist(),
            "encrypted": encrypted_response.hex(),
        })

    @app.route("/predict/raw", methods=["POST"])
    def predict_raw():
        data = request.get_json(force=True)
        raw_values = data.get("pixels")

        if not raw_values or len(raw_values) != 784:
            return jsonify({"error": "Expected 784 pixel values"}), 400

        img = np.array(raw_values, dtype=np.float32).reshape(28, 28) / 255.0
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = int(output.argmax(dim=1).item())

        return jsonify({"prediction": prediction})

    return app


def run_server(model_path: str, host: str = "0.0.0.0", port: int = 5000):
    key = Fernet.generate_key()
    print(f"\nEncryption key (save this): {key.decode()}")
    app = create_app(model_path, key)
    app.run(host=host, port=port, debug=False, ssl_context=None)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/global_model.pt"
    run_server(model_path)
