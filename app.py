"""
APLIKASI WEB FLASK - KLASIFIKASI JENIS KENDARAAN
Menggunakan Model Transfer Learning VGG16
"""

import os
import json
import uuid

import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "vgg16_vehicle_model.h5")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import gdown

GDRIVE_FILE_ID = "1k0z_7Wrn6Q30agHYWITFeAR9L9tU6vEz"

if not os.path.exists(MODEL_PATH):
    print("Model belum ada, mengunduh dari Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    print("Download model selesai.")

print("Memuat model...")
model = load_model(MODEL_PATH)
print("Memuat model...")
model = load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
print("Model siap. Kelas:", idx_to_class)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(filepath):
    img = keras_image.load_img(filepath, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = idx_to_class[predicted_idx]
    confidence = float(predictions[predicted_idx]) * 100

    all_probs = {
        idx_to_class[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(predictions))
    }

    return predicted_class, confidence, all_probs


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filename = secure_filename(unique_filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        predicted_class, confidence, all_probs = predict_image(filepath)

        image_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "result.html",
            image_url=image_url,
            predicted_class=predicted_class,
            confidence=round(confidence, 2),
            all_probs=all_probs,
        )

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)