import torch
import numpy as np
import joblib
import sounddevice as sd
import librosa

from transformers import HubertModel, Wav2Vec2FeatureExtractor

# -----------------------------
# Load saved model
# -----------------------------
model = joblib.load("emotion_model_hubert_svm.pkl")
scaler = joblib.load("scaler_hubert.pkl")
le = joblib.load("label_encoder_hubert.pkl")

# -----------------------------
# Load HuBERT
# -----------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-base-ls960"
)
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hubert.to(device)

# -----------------------------
# SETTINGS
# -----------------------------
SAMPLE_RATE = 16000
DURATION = 3

print("🎤 Emotion Detection System Ready")

while True:
    user_input = input("\nPress ENTER to speak or 'q' to quit: ")

    if user_input.lower() == 'q':
        print("👋 Exiting...")
        break

    print("🎤 Speak now...")

    # Record audio
    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()

    audio = audio.flatten()

    # -----------------------------
    # 🔥 FIX 1: Remove silence
    # -----------------------------
    audio, _ = librosa.effects.trim(audio)

    # -----------------------------
    # 🔥 FIX 2: Avoid empty audio
    # -----------------------------
    if len(audio) == 0:
        print("⚠️ No speech detected")
        continue

    # -----------------------------
    # 🔥 FIX 3: Normalize safely
    # -----------------------------
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # -----------------------------
    # 🔥 FIX 4: Boost volume
    # -----------------------------
    audio = audio * 3
    audio = np.clip(audio, -1, 1)

    # -----------------------------
    # Fix length
    # -----------------------------
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE * DURATION]

    # -----------------------------
    # Feature Extraction
    # -----------------------------
    inputs = feature_extractor(audio,
                               sampling_rate=16000,
                               return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hubert(**inputs)

    hidden = outputs.last_hidden_state

    # Pooling
    mean_pool = torch.mean(hidden, dim=1)
    max_pool = torch.max(hidden, dim=1).values
    embedding = torch.cat((mean_pool, max_pool), dim=1)

    embedding = embedding.cpu().numpy()

    # Scale
    embedding = scaler.transform(embedding)

    # -----------------------------
    # 🔥 PREDICTION WITH CONFIDENCE
    # -----------------------------
    probs = model.predict_proba(embedding)
    pred = np.argmax(probs)
    confidence = np.max(probs)

    emotion = le.inverse_transform([pred])[0]

    # -----------------------------
    # 🔥 FIX 5: Bias correction
    # -----------------------------
    if confidence < 0.5:
        emotion = "neutral"

    print(f"🎯 Emotion: {emotion} (confidence: {confidence:.2f})")