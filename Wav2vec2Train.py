# -*- coding: utf-8 -*-
"""
Improved Speech Emotion Recognition
Fix: All predictions = 'sad'
"""

import os
import torch
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# -----------------------------
# Step 1 — Dataset
# -----------------------------
path = "Project/dataset_speech"

emotion_map = {
    "A": "angry",
    "F": "fear",
    "H": "happy",
    "N": "neutral",
    "S": "sad"
}

file_paths = []
emotions = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[0]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                file_paths.append(os.path.join(root, file))
                emotions.append(emotion)

df = pd.DataFrame({"file_path": file_paths, "emotion": emotions})
# 🔥 Balance dataset (equal samples per emotion)
min_samples = df["emotion"].value_counts().min()

df = df.groupby("emotion").apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

print("Balanced data:\n", df["emotion"].value_counts())

print(df["emotion"].value_counts())

# -----------------------------
# Step 2 — Audio Preprocessing
# -----------------------------
TARGET_SR = 16000
DURATION = 3
MAX_SAMPLES = TARGET_SR * DURATION

audio_data = []
labels = []

for _, row in df.iterrows():
    try:
        audio, _ = librosa.load(row["file_path"], sr=TARGET_SR)

        if len(audio) > MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        else:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

        audio_data.append(audio)
        labels.append(row["emotion"])

    except Exception as e:
        print("Error:", e)

audio_data = np.array(audio_data)
labels = np.array(labels)

# -----------------------------
# Step 3 — Wav2Vec2 Features
# -----------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

features = []

print("Extracting features...")

for audio in audio_data:
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state

    # FIXED pooling (same everywhere)
    mean_pool = torch.mean(hidden, dim=1)
    max_pool = torch.max(hidden, dim=1).values

    embedding = torch.cat((mean_pool, max_pool), dim=1)

    features.append(embedding.cpu().numpy())

features = np.vstack(features)

# -----------------------------
# Step 4 — Encode + Balance
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

# CLASS WEIGHTS (IMPORTANT FIX)
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y),
#     y=y
# )

# -----------------------------
# Step 5 — Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.2,
    stratify=y,   # 🔥 IMPORTANT FIX
    random_state=42
)
import joblib

joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

# -----------------------------
# Step 6 — Feature Scaling (CRITICAL FIX)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 7 — Improved Classifier
# -----------------------------
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

# 🔥 Better model for small dataset
classifier = SVC(kernel='rbf', probability=True)

# 🔥 Apply class balancing during training
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

classifier.fit(X_train, y_train, sample_weight=sample_weights)

# -----------------------------
# Step 8 — Evaluation
# -----------------------------
y_pred = classifier.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# Step 9 — Save Models
# -----------------------------
joblib.dump(classifier, "emotion_modelwav.pkl")
joblib.dump(le, "label_encoderwav.pkl")
joblib.dump(scaler, "scalerwav.pkl")

print("Saved all models!")

# -----------------------------
# Step 10 — Prediction (FIXED)
# -----------------------------
def predict_emotion(file_path):
    audio, _ = librosa.load(file_path, sr=TARGET_SR)

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state

    mean_pool = torch.mean(hidden, dim=1)
    max_pool = torch.max(hidden, dim=1).values
    embedding = torch.cat((mean_pool, max_pool), dim=1).cpu().numpy()

    # 🔥 APPLY SCALER (YOU MISSED THIS BEFORE)
    embedding = scaler.transform(embedding)

    pred = classifier.predict(embedding)

    return le.inverse_transform(pred)[0]

# TEST
sample = df["file_path"].iloc[0]
print("Prediction:", predict_emotion(sample))
print("Predicted labels:", np.unique(y_pred, return_counts=True))

# ==========================================
# 🎤 LIVE MICROPHONE EMOTION DETECTION
# ==========================================

import sounddevice as sd
import queue
from collections import deque
import noisereduce as nr

q = queue.Queue()

# Audio settings
SAMPLERATE = 16000
DURATION = 2   # seconds per chunk

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

print("🎤 Starting Real-Time Emotion Detection... Press Ctrl+C to stop")

emotion_buffer = deque(maxlen=5)

try:
    with sd.InputStream(samplerate=SAMPLERATE,
                        channels=1,
                        callback=audio_callback,
                        blocksize=int(SAMPLERATE * DURATION)):

        while True:
            audio = q.get().flatten()

            # 🔥 Noise reduction
            audio = nr.reduce_noise(y=audio, sr=SAMPLERATE)

            # 🔥 Silence check
            if np.mean(np.abs(audio)) < 0.01:
                print("No speech...")
                continue

            # 🔥 Feature extraction
            inputs = processor(audio, sampling_rate=SAMPLERATE, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            hidden = outputs.last_hidden_state

            mean_pool = torch.mean(hidden, dim=1)
            max_pool = torch.max(hidden, dim=1).values
            embedding = torch.cat((mean_pool, max_pool), dim=1).cpu().numpy().flatten()

            # 🔥 MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLERATE, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)

            final_feature = np.concatenate([embedding, mfcc_mean]).reshape(1, -1)

            final_feature = scaler.transform(final_feature)

            pred = classifier.predict(final_feature)[0]

            # 🔥 Smooth output
            emotion_buffer.append(pred)
            final_pred = max(set(emotion_buffer), key=emotion_buffer.count)

            emotion = le.inverse_transform([final_pred])[0]

            print("🎯 Emotion:", emotion)

except KeyboardInterrupt:
    print("\nStopped!")