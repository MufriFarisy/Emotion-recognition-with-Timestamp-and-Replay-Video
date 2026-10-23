import os
import cv2
import numpy as np

from keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# PATH
# =========================

MODEL_PATH = "models/_mini_XCEPTION.102-0.66.hdf5"
DATASET_PATH = "User Dataset"

# =========================
# LABEL EMOSI
# =========================

EMOTIONS = [
    "angry",
    "disgust",
    "scared",
    "happy",
    "sad",
    "surprised",
    "neutral"
]

# label -> angka
label_to_idx = {
    emotion: idx
    for idx, emotion in enumerate(EMOTIONS)
}

# angka -> label
idx_to_label = {
    idx: emotion
    for idx, emotion in enumerate(EMOTIONS)
}

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded.")

# =========================
# LIST PENYIMPANAN
# =========================

y_true = []
y_pred = []

total_images = 0

# =========================
# LOOP DATASET
# =========================

for emotion_folder in EMOTIONS:

    folder_path = os.path.join(
        DATASET_PATH,
        emotion_folder
    )

    if not os.path.exists(folder_path):
        print(f"Folder tidak ditemukan: {folder_path}")
        continue

    print(f"\nMembaca kelas: {emotion_folder}")

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            folder_path,
            filename
        )

        # =========================
        # BACA GAMBAR
        # =========================

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(
                f"Gagal membaca {image_path}"
            )
            continue

        # =========================
        # PREPROCESS
        # =========================

        image = cv2.resize(
            image,
            (64, 64)
        )

        image = image.astype(
            "float32"
        ) / 255.0

        image = np.expand_dims(
            image,
            axis=-1
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        # =========================
        # PREDIKSI
        # =========================

        prediction = model.predict(
            image,
            verbose=0
        )[0]

        predicted_idx = np.argmax(
            prediction
        )

        predicted_label = (
            idx_to_label[
                predicted_idx
            ]
        )

        # =========================
        # GROUND TRUTH
        # =========================

        true_idx = (
            label_to_idx[
                emotion_folder
            ]
        )

        y_true.append(
            true_idx
        )

        y_pred.append(
            predicted_idx
        )

        total_images += 1

# =========================
# HASIL EVALUASI
# =========================

print("\n==========================")
print("HASIL EVALUASI")
print("==========================")

print(
    f"Total gambar: {total_images}"
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-Score : {f1:.4f}"
)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=EMOTIONS,
        zero_division=0
    )
)