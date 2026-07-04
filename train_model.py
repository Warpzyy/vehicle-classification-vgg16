import os
import json
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# =========================================================================
# 1. KONFIGURASI
# =========================================================================
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16          # dikecilkan karena dataset kita kecil (100/kelas)
EPOCHS = 15
MODEL_OUTPUT_DIR = "model"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")

# =========================================================================
# 2. PERSIAPAN DATASET (dengan augmentasi untuk data training)
# =========================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True,
)

val_data = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False,
)

test_data = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False,
)

num_classes = len(train_data.class_indices)
print("Jumlah kelas terdeteksi:", num_classes)
print("Kelas:", train_data.class_indices)

with open(os.path.join(MODEL_OUTPUT_DIR, "class_indices.json"), "w") as f:
    json.dump(train_data.class_indices, f, indent=2)

# =========================================================================
# 3. LOAD MODEL VGG16 PRETRAINED (tanpa Fully Connected Layer)
# =========================================================================
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

# =========================================================================
# 4. TAMBAHKAN LAYER KLASIFIKASI BARU
# =========================================================================
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# =========================================================================
# 5. TAHAP 1: TRAINING (FEATURE EXTRACTION)
# =========================================================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint(
        os.path.join(MODEL_OUTPUT_DIR, "vgg16_vehicle_model.h5"),
        monitor="val_accuracy", save_best_only=True,
    ),
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# =========================================================================
# 6. TAHAP 2: FINE-TUNING
# =========================================================================
print("\n=== Mulai Fine-Tuning ===")
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks,
)

# =========================================================================
# 7. EVALUASI MODEL PADA DATA TEST
# =========================================================================
test_loss, test_acc = model.evaluate(test_data)
print(f"\nAkurasi Model pada Data Test: {test_acc:.4f}")
print(f"Loss Model pada Data Test: {test_loss:.4f}")

model.save(os.path.join(MODEL_OUTPUT_DIR, "vgg16_vehicle_model.h5"))
print("Model tersimpan di:", os.path.join(MODEL_OUTPUT_DIR, "vgg16_vehicle_model.h5"))

# =========================================================================
# 8. VISUALISASI HASIL TRAINING
# =========================================================================
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Akurasi Model")
plt.xlabel("Epoch")
plt.ylabel("Akurasi")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Loss Model")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("hasil_training.png")
print("Grafik hasil training tersimpan di: hasil_training.png")
plt.show()