# 🎤 Speech Emotion Detection using HuBERT & Wav2Vec2 + SVM

## 📌 Overview

This project focuses on detecting human emotions from speech signals using advanced deep learning feature extraction models and machine learning classification.

Two state-of-the-art models are used:

* 🔹 Wav2Vec2
* 🔹 HuBERT

These models extract deep speech features, which are then classified using a **Support Vector Machine (SVM)** to identify emotions.

---

## 🎯 Objectives

* Convert speech signals into meaningful features
* Compare Wav2Vec2 and HuBERT performance
* Build a real-time emotion detection system
* Apply signal processing concepts (sampling, pitch, energy)
* Visualize model performance

---

## 🧠 Emotions Detected

* 😄 Happy
* 😢 Sad
* 😡 Angry
* 😨 Fear
* 😐 Neutral

---

## ⚙️ Technologies Used

* Python
* NumPy, Pandas
* Librosa
* PyTorch
* HuggingFace Transformers
* Scikit-learn (SVM)
* Matplotlib & Seaborn
* Streamlit (UI Dashboard)

---

## 📂 Project Structure

```text id="struct001"
├── train_wav2vec2.py
├── train_hubert_svm.py
├── evaluate_graphs.py
├── real_time_hubert.py
├── ui.py
│
├── emotion_model_wav2vec2.pkl
├── emotion_model_hubert_svm.pkl
├── scaler_hubert.pkl
├── label_encoder_hubert.pkl
├── X_test_hubert.pkl
├── y_test_hubert.pkl
│
└── dataset/
```

---

## 📊 Dataset Information
📂 Dataset used in this project is available here: [Google Drive Link](https://drive.google.com/file/d/1XLhll_ZNXSZWNUaDmEtIq8RE9aBRWOXq/view?usp=sharing)

The dataset consists of **600 audio samples** collected from **8 speakers** across **5 emotions**.

* Format: WAV
* Sampling Rate: 16 kHz
* Duration: 2–3 seconds
* Channels: Mono

### Emotion Encoding:

| Code | Emotion |
| ---- | ------- |
| A    | Angry   |
| F    | Fear    |
| H    | Happy   |
| N    | Neutral |
| S    | Sad     |

Example:

```
A-01.wav → Angry
H-02.wav → Happy
```

---

## 📊 Model Performance

### 🔹 Wav2Vec2 + SVM

* Accuracy: **89.17%**
* F1-score: **0.89**

Class-wise:

* Angry: 0.96
* Fear: 0.78
* Happy: 0.90
* Neutral: 0.96
* Sad: 0.86

👉 Performs well but struggles with fear & sad

---

### 🔹 HuBERT + SVM

* Accuracy: **95%**
* F1-score: **0.95**

Class-wise:

* Angry: 0.94
* Fear: 0.94
* Happy: 0.96
* Neutral: 0.98
* Sad: 0.93

👉 More balanced and accurate

---

## 📊 Model Comparison

| Metric     | HuBERT + SVM | Wav2Vec2 + SVM |
| ---------- | ------------ | -------------- |
| Accuracy   | **95% ✅**    | 89.17%         |
| F1-score   | **0.95**     | 0.89           |
| Stability  | High         | Moderate       |
| Weak Class | Slight (Sad) | Fear & Sad     |
| Overall    | ⭐ Best       | Good           |

👉 HuBERT performs better due to stronger feature representation 

---

## ▶️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/speech-emotion-detection.git
cd speech-emotion-detection
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install numpy pandas librosa scikit-learn matplotlib seaborn
pip install torch transformers sounddevice streamlit
```

---

## ▶️ How to Run

### 🔹 Train Models

```bash
python train_wav2vec2.py
python train_hubert_svm.py
```

### 🔹 Generate Graphs

```bash
python evaluate_graphs.py
```

### 🔹 Real-Time Detection

```bash
python real_time_hubert.py
```

### 🔹 Run UI Dashboard

```bash
streamlit run ui.py
```

---

## 📊 Output

* Confusion Matrix
* Precision / Recall / F1-score graphs
* Real-time emotion detection
* Interactive dashboard

---

## 🔬 Key Concepts Used

* Sampling (16 kHz)
* Nyquist Theorem
* Feature Extraction (HuBERT, Wav2Vec2)
* Pitch & Energy Analysis
* Machine Learning (SVM)

---

## 🚀 Applications

* 📞 Call Center Emotion Monitoring
* 🏥 Mental Health Analysis
* 🤖 AI Assistants
* 🎧 Customer Feedback Systems

---

## ⚠️ Limitations

* Real-time accuracy affected by noise
* Dataset may not represent real-world diversity
* Subtle emotions are harder to classify

---

## 🔮 Future Work

* Fine-tune HuBERT model
* Add noise robustness
* Deploy as web application
* Integrate with live call systems

---

## 👩‍💻 Author

**Sakshi Gaikwad**
Electronics & Telecommunication Engineering

---

## ⭐ Conclusion

This project demonstrates how signal processing and deep learning can be combined to detect human emotions from speech. HuBERT-based feature extraction significantly improves performance compared to Wav2Vec2, making the system more robust and reliable.

