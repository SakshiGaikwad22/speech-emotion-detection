# realtime_emotion.py

import sounddevice as sd
import numpy as np
import librosa
import torch
import joblib
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# -----------------------------
# SETTINGS
# -----------------------------
SAMPLE_RATE = 16000
DURATION = 3  # seconds
MAX_SAMPLES = SAMPLE_RATE * DURATION

# -----------------------------
# LOAD MODELS
# -----------------------------

print("Loading models...")

classifier = joblib.load("emotion_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec_model.to(device)

print("✅ Ready!\n")

# -----------------------------
# RECORD AUDIO
# -----------------------------
def record_audio():
    print("🎤 Speak now...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print("⏳ Processing...")
    return audio.flatten()

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(audio):
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

    hidden = outputs.last_hidden_state

    mean_pool = torch.mean(hidden, dim=1)
    max_pool = torch.max(hidden, dim=1).values

    embedding = torch.cat((mean_pool, max_pool), dim=1).cpu().numpy()

    # VERY IMPORTANT
    embedding = scaler.transform(embedding)

    return embedding

# -----------------------------
# PREDICT
# -----------------------------
def predict_emotion(audio):
    features = extract_features(audio)
    pred = classifier.predict(features)
    emotion = label_encoder.inverse_transform(pred)
    return emotion[0]

# -----------------------------
# MAIN LOOP
# -----------------------------
import speech_recognition as sr

def recognize_speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Speak now...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("📝 You said:", text)
        return text.lower()
    except:
        print("❌ Could not understand")
        return ""

def keyword_emotion(text):
    if "happy" in text:
        return "happy"
    elif "sad" in text:
        return "sad"
    elif "angry" in text:
        return "angry"
    elif "fear" in text:
        return "fear"
    elif "neutral" in text:
        return "neutral"
    else:
        return "unknown"

# -----------------------------
# MAIN LOOP (HARDCODED DEMO)
# -----------------------------
while True:
    text = recognize_speech()
    emotion = keyword_emotion(text)

    print(f"🎯 Predicted Emotion: {emotion.upper()}\n")

    cont = input("Press Enter to continue or type 'q' to quit: ")
    if cont.lower() == 'q':
        break