"""
SPLIT DATASET - dari dataset_raw/{mobil,motor,bus,truk} ke dataset/train|val|test
"""

import os
import shutil
import random

SRC_DIR = "dataset_raw"
DEST_DIR = "dataset"
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(RANDOM_SEED)


def main():
    if not os.path.isdir(SRC_DIR):
        print(f"[ERROR] Folder '{SRC_DIR}' tidak ditemukan.")
        return

    classes = [
        d for d in os.listdir(SRC_DIR)
        if os.path.isdir(os.path.join(SRC_DIR, d))
    ]

    if not classes:
        print(f"[ERROR] Tidak ada subfolder kelas di dalam '{SRC_DIR}'.")
        return

    print(f"Kelas terdeteksi: {classes}\n")

    for split in SPLIT_RATIO:
        for cls in classes:
            os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

    total_summary = []

    for cls in classes:
        src_class_dir = os.path.join(SRC_DIR, cls)
        files = [
            f for f in os.listdir(src_class_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        if len(files) == 0:
            print(f"[PERINGATAN] Folder '{cls}' kosong, dilewati.")
            continue

        random.shuffle(files)
        n = len(files)
        n_train = int(n * SPLIT_RATIO["train"])
        n_val = int(n * SPLIT_RATIO["val"])

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dest_dir = os.path.join(DEST_DIR, split_name, cls)
            for fname in split_files:
                shutil.copy(
                    os.path.join(src_class_dir, fname),
                    os.path.join(dest_dir, fname),
                )

        total_summary.append(
            (cls, n, len(splits["train"]), len(splits["val"]), len(splits["test"]))
        )

    print(f"{'Kelas':<10} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 40)
    for cls, total, tr, va, te in total_summary:
        print(f"{cls:<10} {total:>6} {tr:>6} {va:>6} {te:>6}")

    if any(total < 50 for _, total, *_ in total_summary):
        print("\n⚠️  Ada kelas dengan gambar kurang dari 50. Model mungkin kurang akurat.")

    print(f"\n✅ Selesai! Dataset siap dipakai di folder '{DEST_DIR}/'")


if __name__ == "__main__":
    main()