import torch
import torch.nn as nn
import torchaudio
import time
import os
from typing import Tuple

# Константи
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512


class SmallCNN(nn.Module):
    """Проста CNN модель для класифікації аудіо команд"""
    
    def __init__(self, n_classes: int = 4):
        super(SmallCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Перший блок
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Другий блок
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Третій блок
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((16, 32))  # Фіксований розмір виходу
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def wav_to_melspec(waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Перетворює аудіо хвилю в Mel-спектрограму"""
    # Resample якщо потрібно
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Mel-спектрограма
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # Перетворення в dB
    mel_spec = mel_transform(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return mel_spec_db


def measure_latency(model: nn.Module, example_input: torch.Tensor, runs: int = 30) -> float:
    """Вимірює середню латентність інференсу моделі"""
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(example_input)
    
    # Вимірювання
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(example_input)
            end = time.time()
            times.append((end - start) * 1000)  # в мілісекундах
    
    return sum(times) / len(times)


def save_model(model: nn.Module, path: str) -> int:
    """Зберігає state_dict моделі та повертає розмір файлу"""
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)


def export_torchscript(model: nn.Module, example_input: torch.Tensor, path: str) -> int:
    """Експортує модель в TorchScript та повертає розмір файлу"""
    model.eval()
    scripted_model = torch.jit.trace(model, example_input)
    torch.jit.save(scripted_model, path)
    return os.path.getsize(path)


def load_model(model_class, path: str, n_classes: int = 4, device: torch.device = None) -> nn.Module:
    """Завантажує модель з збереженого state_dict"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class(n_classes=n_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model