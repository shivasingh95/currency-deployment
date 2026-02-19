from flask import Flask, request, jsonify
import cv2
import numpy as np
from model_utils import predict_from_image

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Currency Classification API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    label, confidence = predict_from_image(image)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
