# train.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
from model import CNNLSTMEmotionModel
from tqdm import tqdm

class EmotionDataset(Dataset):
    def __init__(self, data_dir, labels_map, transform=None):
        self.files = []
        for emotion, label in labels_map.items():
            emotion_path = os.path.join(data_dir, emotion)
            for f in os.listdir(emotion_path):
                if f.endswith('.wav'):
                    self.files.append((os.path.join(emotion_path, f), label))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        waveform, sr = torchaudio.load(path)
        mel = torchaudio.transforms.MelSpectrogram(sr)(waveform)
        if self.transform:
            mel = self.transform(mel)
        return mel, label

def train_model(data_dir="data/", output_dir="outputs/", epochs=10, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_map = {"angry":0,"happy":1,"sad":2,"neutral":3}
    dataset = EmotionDataset(data_dir, labels_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNNLSTMEmotionModel(num_classes=len(labels_map)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for mel, label in tqdm(loader, desc=f"Epoch {epoch+1}"):
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(mel)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
    print("✅ Model saved!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="outputs/")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train_model(args.data_dir, args.output_dir, args.epochs)
