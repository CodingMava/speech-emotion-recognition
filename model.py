# model.py
import torch
import torch.nn as nn
import torchaudio

class CNNLSTMEmotionModel(nn.Module):
    """
    Simple CNN + LSTM model for speech emotion classification.
    """
    def __init__(self, num_classes=8):
        super(CNNLSTMEmotionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(32 * 32, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.view(b, c * f, t).permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out
