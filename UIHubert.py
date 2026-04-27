import tkinter as tk
import threading
import torch
import numpy as np
import joblib
import sounddevice as sd
import librosa

from transformers import HubertModel, Wav2Vec2FeatureExtractor

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
model = joblib.load("emotion_model_hubert_svm.pkl")
scaler = joblib.load("scaler_hubert.pkl")
le = joblib.load("label_encoder_hubert.pkl")

# -----------------------------
# LOAD HUBERT MODEL
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

# -----------------------------
# REAL EMOTION DETECTION
# -----------------------------
def detect_emotion():
    status_label.config(text="🎤 Listening...")
    root.update()

    # Record audio
    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()

    audio = audio.flatten()

    # Remove silence
    audio, _ = librosa.effects.trim(audio)

    if len(audio) == 0:
        return "No Speech ❌", "red"

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Boost volume
    audio = audio * 3
    audio = np.clip(audio, -1, 1)

    # Fix length
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE * DURATION]

    # Feature extraction
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

    # Prediction
    probs = model.predict_proba(embedding)
    pred = np.argmax(probs)
    confidence = np.max(probs)

    emotion = le.inverse_transform([pred])[0]

    # Low confidence → neutral
    if confidence < 0.5:
        emotion = "neutral"

    # UI Colors
    color_map = {
        "happy": "#22c55e",
        "sad": "#3b82f6",
        "angry": "#ef4444",
        "fear": "#a855f7",
        "neutral": "#94a3b8"
    }

    color = color_map.get(emotion.lower(), "#facc15")

    return f"{emotion.upper()} ({confidence:.2f})", color


# -----------------------------
# BUTTON ACTIONS
# -----------------------------
def start_detection():
    emotion, color = detect_emotion()
    result_label.config(text=emotion, fg=color)
    status_label.config(text="Done ✅")

def threaded_start():
    threading.Thread(target=start_detection).start()

def reset_ui():
    result_label.config(text="---", fg="#22c55e")
    status_label.config(text="Tap to Record")

# -----------------------------
# MIC EVENTS
# -----------------------------
def on_mic_click(event):
    reset_ui()
    threaded_start()

def on_hover(event):
    mic_canvas.itemconfig(circle, fill="#4f46e5")

def on_leave(event):
    mic_canvas.itemconfig(circle, fill="#6366f1")

# -----------------------------
# UI DESIGN
# -----------------------------
root = tk.Tk()
root.title("Emotion Detector AI")
root.geometry("450x600")
root.configure(bg="#0b1020")

main_frame = tk.Frame(root, bg="#0b1020")
main_frame.pack(expand=True)

# Title
title = tk.Label(main_frame, text="AI Emotion Detector",
                 font=("Segoe UI", 24, "bold"),
                 fg="#a78bfa", bg="#0b1020")
title.pack(pady=20)

subtitle = tk.Label(main_frame, text="Real-time Voice Emotion Recognition",
                    font=("Segoe UI", 11),
                    fg="#9ca3af", bg="#0b1020")
subtitle.pack()

# MIC BUTTON
mic_canvas = tk.Canvas(main_frame, width=140, height=140,
                       bg="#0b1020", highlightthickness=0)
mic_canvas.pack(pady=40)

circle = mic_canvas.create_oval(10, 10, 130, 130,
                               fill="#6366f1", outline="")

mic_icon = mic_canvas.create_text(70, 70, text="🎤",
                                 font=("Segoe UI", 30),
                                 fill="white")

mic_canvas.bind("<Button-1>", on_mic_click)
mic_canvas.bind("<Enter>", on_hover)
mic_canvas.bind("<Leave>", on_leave)

# Status
status_label = tk.Label(main_frame, text="Tap to Record",
                        font=("Segoe UI", 11),
                        fg="#cbd5f5", bg="#0b1020")
status_label.pack(pady=10)

# Result Box
result_frame = tk.Frame(main_frame, bg="#111827")
result_frame.pack(pady=30, ipadx=40, ipady=20)

result_title = tk.Label(result_frame, text="Detected Emotion",
                        font=("Segoe UI", 12),
                        fg="#9ca3af", bg="#111827")
result_title.pack()

result_label = tk.Label(result_frame, text="---",
                        font=("Segoe UI", 26, "bold"),
                        fg="#22c55e", bg="#111827")
result_label.pack(pady=10)

# Reset Button
reset_btn = tk.Button(main_frame,
                      text="🔄 Record Again",
                      command=reset_ui,
                      font=("Segoe UI", 11),
                      bg="#374151",
                      fg="white",
                      bd=0,
                      padx=15,
                      pady=5)
reset_btn.pack(pady=10)

# Footer
footer = tk.Label(root, text="HuBERT + SVM Emotion Recognition",
                  font=("Segoe UI", 9),
                  fg="#6b7280", bg="#0b1020")
footer.pack(side="bottom", pady=10)

root.mainloop()