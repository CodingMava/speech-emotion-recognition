# app.py
from fastapi import FastAPI, UploadFile, File
import torchaudio
import torch
from model import CNNLSTMEmotionModel

app = FastAPI(title="Speech Emotion Detection API")

# Load model
labels = ["angry", "happy", "sad", "neutral"]
model = CNNLSTMEmotionModel(num_classes=len(labels))
model.load_state_dict(torch.load("outputs/best_model.pt", map_location="cpu"))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    waveform, sr = torchaudio.load(file.file)
    mel = torchaudio.transforms.MelSpectrogram(sr)(waveform)
    with torch.no_grad():
        logits = model(mel)
        emotion = labels[int(torch.argmax(logits))]
    return {"predicted_emotion": emotion}
