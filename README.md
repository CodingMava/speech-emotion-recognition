# 🎙️ AI Project 3: Speech Emotion Detection

This project detects **human emotions** (angry, happy, sad, neutral) from voice recordings using a CNN-LSTM model trained on Mel spectrogram features.

---

## 🚀 Features
- Converts audio `.wav` to Mel spectrogram
- CNN + LSTM hybrid model
- FastAPI service for inference
- Dockerized for deployment

---

## ⚙️ Setup
```bash
pip install -r requirements.txt
python train.py --data_dir data/ --epochs 5
uvicorn app:app --reload
