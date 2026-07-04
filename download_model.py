"""
DOWNLOAD MODEL DARI GOOGLE DRIVE
Dipakai karena file model (.h5) terlalu besar untuk disimpan di GitHub.

CARA PAKAI:
    python download_model.py
"""

import os
import gdown

FILE_ID = "1k0z_7Wrn6Q30agHYWITFeAR9L9tU6vEz"
OUTPUT_PATH = "model/vgg16_vehicle_model.h5"

os.makedirs("model", exist_ok=True)

if os.path.exists(OUTPUT_PATH):
    print(f"Model sudah ada di {OUTPUT_PATH}, skip download.")
else:
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Mengunduh model dari Google Drive...")
    gdown.download(url, OUTPUT_PATH, quiet=False)
    print("Selesai!")