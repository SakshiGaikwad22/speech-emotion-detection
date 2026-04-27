import os
import torch
import numpy as np
import pandas as pd
import librosa
import joblib

from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Dataset Loading
# -----------------------------
path = "archive/dataset_speech"

emotion_map = {
    "A": "angry",
    "F": "fear",
    "H": "happy",
    "N": "neutral",
    "S": "sad"
}

files, emotions = [], []

for root, _, fs in os.walk(path):
    for f in fs:
        if f.endswith(".wav"):
            code = f.split("-")[0]
            emotion = emotion_map.get(code)
            if emotion:
                files.append(os.path.join(root, f))
                emotions.append(emotion)

df = pd.DataFrame({"file": files, "emotion": emotions})

# Balance dataset
min_samples = df["emotion"].value_counts().min()
df = df.groupby("emotion").apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

# -----------------------------
# Audio Processing
# -----------------------------
TARGET_SR = 16000
MAX_LEN = TARGET_SR * 3

audio_data, labels = [], []

for _, row in df.iterrows():
    audio, _ = librosa.load(row["file"], sr=TARGET_SR)

    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))

    audio_data.append(audio)
    labels.append(row["emotion"])

# -----------------------------
# HuBERT Feature Extraction
# -----------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hubert.to(device)

features = []

print("Extracting HuBERT features...")

for audio in audio_data:
    inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hubert(**inputs)

    hidden = outputs.last_hidden_state

    mean_pool = torch.mean(hidden, dim=1)
    max_pool = torch.max(hidden, dim=1).values

    embedding = torch.cat((mean_pool, max_pool), dim=1)
    features.append(embedding.cpu().numpy())

X = np.vstack(features)

# -----------------------------
# Labels
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# SVM MODEL (Improved)
# -----------------------------
classifier = SVC(kernel='rbf', C=10, gamma='scale', probability=True)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

classifier.fit(X_train, y_train, sample_weight=sample_weights)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# Save (NEW NAMES)
# -----------------------------
joblib.dump(classifier, "emotion_model_hubert_svm.pkl")
joblib.dump(scaler, "scaler_hubert.pkl")
joblib.dump(le, "label_encoder_hubert.pkl")

joblib.dump(X_test, "X_test_hubert.pkl")
joblib.dump(y_test, "y_test_hubert.pkl")

print("Model saved successfully!")