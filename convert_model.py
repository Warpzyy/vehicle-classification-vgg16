"""
Convert model dari format .h5 (legacy) ke format .keras (native, lebih robust)
Jalankan SEKALI SAJA di komputer yang model-nya sudah terbukti bisa di-load.
"""

from tensorflow.keras.models import load_model

print("Memuat model dari format .h5...")
model = load_model("model/vgg16_vehicle_model.h5")

print("Menyimpan ulang dalam format .keras...")
model.save("model/vgg16_vehicle_model.keras")

print("Selesai! File baru: model/vgg16_vehicle_model.keras")