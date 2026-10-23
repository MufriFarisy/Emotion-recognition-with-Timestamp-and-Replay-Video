from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os
import sys
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# ==========================
# PATH SETUP
# ==========================
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")

if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

detection_model_path = os.path.join(
    base_dir,
    "haarcascade_files",
    "haarcascade_frontalface_default.xml"
)

emotion_model_path = os.path.join(
    base_dir,
    "models",
    "_mini_XCEPTION.102-0.66.hdf5"
)

# ==========================
# LOAD MODEL
# ==========================
face_detection = cv2.CascadeClassifier(detection_model_path)

emotion_classifier = load_model(
    emotion_model_path,
    compile=False
)

EMOTIONS = [
    "angry",
    "disgust",
    "scared",
    "happy",
    "sad",
    "surprised",
    "neutral"
]

# ==========================
# SAVE FOLDER
# ==========================
save_dir = "saved_frames"
os.makedirs(save_dir, exist_ok=True)

folder_index = 1
image_count = 0

current_folder = os.path.join(
    save_dir,
    f"batch_{folder_index}"
)

os.makedirs(current_folder, exist_ok=True)

# ==========================
# CSV SETUP
# ==========================
csv_file = os.path.join(
    save_dir,
    "hasil_evaluasi.csv"
)

if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "timestamp",
            "true_label",
            "predicted_label",
            "angry",
            "disgust",
            "scared",
            "happy",
            "sad",
            "surprised",
            "neutral"
        ])
# ==========================
# VIDEO CAPTURE
# ==========================
cv2.namedWindow("your_face")

camera = cv2.VideoCapture(0)

width = 640
height = 480

camera.set(3, width)
camera.set(4, height)

# ==========================
# VIDEO WRITER
# ==========================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

start_video_time = time.time()

video_name = datetime.now().strftime(
    "%Y_%m_%d_%H_%M_%S"
)

video_writer = cv2.VideoWriter(
    os.path.join(
        current_folder,
        f"{video_name}.mp4"
    ),
    fourcc,
    20.0,
    (width, height)
)

# ==========================
# FPS
# ==========================
prev_time = time.time()

# ==========================
# GROUND TRUTH
# ==========================
ground_truth = "neutral"

key_map = {
    ord("1"): "angry",
    ord("2"): "disgust",
    ord("3"): "scared",
    ord("4"): "happy",
    ord("5"): "sad",
    ord("6"): "surprised",
    ord("7"): "neutral"
}

# ==========================
# DATA GRAFIK
# ==========================

frame_counter = 0

time_history = []

emotion_history = {
    "angry": [],
    "disgust": [],
    "scared": [],
    "happy": [],
    "sad": [],
    "surprised": [],
    "neutral": []
}

# ==========================
# LOOP
# ==========================
while True:

    grabbed, frame = camera.read()

    if not grabbed:
        break

    frame = imutils.resize(
        frame,
        width=width
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    canvas = np.zeros(
        (250, 300, 3),
        dtype="uint8"
    )

    frameClone = frame.copy()

    # ==========================
    # TIMESTAMP
    # ==========================
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cv2.putText(
        frameClone,
        timestamp,
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    # ==========================
    # FPS
    # ==========================
    curr_time = time.time()

    fps = 1 / (curr_time - prev_time)

    prev_time = curr_time

    cv2.putText(
        frameClone,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ==========================
    # GROUND TRUTH DISPLAY
    # ==========================
    cv2.putText(
        frameClone,
        f"GT: {ground_truth}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ==========================
    # FACE DETECTION
    # ==========================
    if len(faces) > 0:

        faces = sorted(
            faces,
            reverse=True,
            key=lambda x:
            (x[2] - x[0]) *
            (x[3] - x[1])
        )[0]

        (fX, fY, fW, fH) = faces

        roi = gray[
            fY:fY + fH,
            fX:fX + fW
        ]

        roi = cv2.resize(
            roi,
            (64, 64)
        )

        roi = roi.astype("float") / 255.0

        roi = img_to_array(roi)

        roi = np.expand_dims(
            roi,
            axis=0
        )

        preds = emotion_classifier.predict(
            roi,
            verbose=0
        )[0]

        frame_counter += 1
        
        # === APPEND DATA SETIAP BEBERAPA FRAME ===
        if frame_counter % 3 == 0:          # ubah 3 jadi 5 kalau terlalu padat
            current_time = frame_counter / max(fps, 1.0)
            time_history.append(current_time)
            
            for i, emotion in enumerate(EMOTIONS):
                emotion_history[emotion].append(preds[i])

        angry_score = preds[0]
        disgust_score = preds[1]
        scared_score = preds[2]
        happy_score = preds[3]
        sad_score = preds[4]
        surprised_score = preds[5]
        neutral_score = preds[6]

        label = EMOTIONS[
            preds.argmax()
        ]

        # ==========================
        # PROBABILITIES
        # ==========================
        for (
            i,
            (emotion, prob)
        ) in enumerate(
            zip(
                EMOTIONS,
                preds
            )
        ):

            text = (
                f"{emotion}: "
                f"{prob*100:.2f}%"
            )

            w = int(prob * 300)

            cv2.rectangle(
                canvas,
                (7, (i * 35) + 5),
                (w, (i * 35) + 35),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                canvas,
                text,
                (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2
            )

        # ==========================
        # LABEL
        # ==========================
        cv2.putText(
            frameClone,
            label,
            (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            frameClone,
            (fX, fY),
            (fX + fW, fY + fH),
            (0, 255, 0),
            2
        )

        # ==========================
        # SAVE IMAGE
        # ==========================
        image_filename = (
            datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"
            )
            + "_"
            + label
            + ".jpg"
        )

        image_path = os.path.join(
            current_folder,
            image_filename
        )

        cv2.imwrite(
            image_path,
            frameClone
        )

        # ==========================
        # SAVE CSV
        # ==========================
        with open(
            csv_file,
            mode="a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                image_filename,
                timestamp,
                ground_truth,
                label,
                preds[0],
                preds[1],
                preds[2],
                preds[3],
                preds[4],
                preds[5],
                preds[6]
            ])

        image_count += 1

        # ==========================
        # NEW BATCH
        # ==========================
        if image_count == 20:

            folder_index += 1

            current_folder = os.path.join(
                save_dir,
                f"batch_{folder_index}"
            )

            os.makedirs(
                current_folder,
                exist_ok=True
            )

            image_count = 0

    # ==========================
    # SAVE VIDEO
    # ==========================
    video_writer.write(
        frameClone
    )

    # rotate every 5 sec
    if time.time() - start_video_time >= 5:

        video_writer.release()

        video_name = datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S"
        )

        video_writer = cv2.VideoWriter(
            os.path.join(
                current_folder,
                f"{video_name}.mp4"
            ),
            fourcc,
            20.0,
            (width, height)
        )

        start_video_time = time.time()

    # ==========================
    # SHOW
    # ==========================
    cv2.imshow(
        "your_face",
        frameClone
    )

    cv2.imshow(
        "Probabilities",
        canvas
    )
    # ==========================
    # KEYBOARD
    # ==========================
    key = cv2.waitKey(1) & 0xFF

    if key in key_map:
        ground_truth = key_map[key]

    elif key == ord("q"):
        break

# =====================================================
# GRAFIK PERUBAHAN EMOSI
# =====================================================

if len(time_history) > 5:
    plt.figure(figsize=(15, 8))
    
    colors = ['red', 'darkorange', 'limegreen', 'blue', 'purple', 'brown', 'gray']
    
    # Plot semua emosi
    for i, emotion in enumerate(EMOTIONS):
        if len(emotion_history[emotion]) == len(time_history):
            plt.plot(time_history, emotion_history[emotion], 
                    label=emotion.capitalize(), 
                    color=colors[i], 
                    linewidth=2.0, alpha=0.85)
    
    # Dominant Emotion
    dominant_probs = np.array([emotion_history[e] for e in EMOTIONS]).T
    dominant_scores = np.max(dominant_probs, axis=1)
    
    plt.plot(time_history, dominant_scores, 
            label="Dominant Emotion", 
            color='black', linewidth=4.0, alpha=0.95)
    
    plt.xlabel("Waktu (detik)", fontsize=13)
    plt.ylabel("Confidence Score", fontsize=13)
    plt.title("Perubahan Emosi Real-time\n(Mulai dari Neutral → Puncak Emosi)", 
              fontsize=15, pad=20)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xlim(left=0)
    plt.tight_layout()
    
    plt.savefig("grafik_perubahan_emosi_final.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Grafik berhasil disimpan sebagai 'grafik_perubahan_emosi_final.png'")
else:
    print("❌ Data masih terlalu sedikit untuk membuat grafik.")

# ==========================
# CLOSE
# ==========================
camera.release()

video_writer.release()

cv2.destroyAllWindows()