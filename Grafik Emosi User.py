from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================
# PATH SETUP
# ==========================
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

detection_model_path = os.path.join(base_dir, "haarcascade_files", "haarcascade_frontalface_default.xml")
emotion_model_path = os.path.join(base_dir, "models", "_mini_XCEPTION.102-0.66.hdf5")

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# ==========================
# FOLDER VIDEO (UBAH DI SINI)
# ==========================
video_folder = "Actor_04"   # ← Ganti dengan folder videomu

# Ambil semua file video di folder
video_files = [f for f in os.listdir(video_folder) 
               if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# Urutkan berdasarkan nama (karena nama pakai timestamp, jadi otomatis terurut waktu)
video_files.sort()

print(f"✅ Ditemukan {len(video_files)} file video:")
for v in video_files:
    print(f"   • {v}")

# ==========================
# OUTPUT FOLDER
# ==========================
save_dir = "hasil_analisis"
os.makedirs(save_dir, exist_ok=True)

# ==========================
# PROSES SEMUA VIDEO
# ==========================
for video_name in video_files:
    video_path = os.path.join(video_folder, video_name)
    print(f"\n🚀 Memproses: {video_name}")
    
    camera = cv2.VideoCapture(video_path)
    if not camera.isOpened():
        print(f"❌ Gagal membuka {video_name}")
        continue

    fps = camera.get(cv2.CAP_PROP_FPS) or 30
    frame_counter = 0
    time_history = []
    emotion_history = {emotion: [] for emotion in EMOTIONS}

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        frameClone = frame.copy()

        if len(faces) > 0:
            # Ambil wajah terbesar
            faces = sorted(faces, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0]
            (fX, fY, fW, fH) = faces

            roi = gray[fY:fY+fH, fX:fX+fW]
            roi = cv2.resize(roi, (64, 64)).astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi, verbose=0)[0]

            frame_counter += 1

            if frame_counter % 5 == 0:          # sampling rate
                current_time = frame_counter / fps
                time_history.append(current_time)
                for i, emotion in enumerate(EMOTIONS):
                    emotion_history[emotion].append(preds[i])

            label = EMOTIONS[preds.argmax()]
            cv2.putText(frameClone, label, (fX, fY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0,255,0), 2)

        cv2.imshow(f"Processing: {video_name}", frameClone)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    # ==========================
    # BUAT GRAFIK
    # ==========================
    if len(time_history) > 10:
        # Smoothing
        for emotion in EMOTIONS:
            emotion_history[emotion] = pd.Series(emotion_history[emotion]).rolling(window=5, min_periods=1).mean().tolist()

        plt.figure(figsize=(15, 8))
        colors = ['red', 'darkorange', 'limegreen', 'blue', 'purple', 'brown', 'gray']

        for i, emotion in enumerate(EMOTIONS):
            plt.plot(time_history, emotion_history[emotion], 
                    label=emotion.capitalize(), color=colors[i], linewidth=2, alpha=0.85)

        dominant = np.max([emotion_history[e] for e in EMOTIONS], axis=0)
        plt.plot(time_history, dominant, label="Dominant Emotion", color='black', linewidth=4, alpha=0.95)

        plt.title(f"Perubahan Emosi - {video_name}", fontsize=15, pad=20)
        plt.xlabel("Waktu (detik)")
        plt.ylabel("Confidence Score")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        output_path = os.path.join(save_dir, f"grafik_{video_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Grafik selesai: grafik_{video_name}.png")

print("\n🎉 Semua video telah diproses!")