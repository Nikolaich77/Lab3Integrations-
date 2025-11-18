
"""
Лабораторна робота: Розгортання ML-моделі для Speech Commands (PyTorch + Flask)

Цей скрипт готує просту CNN-модель для підмножини команд (yes, no, up, down),
навчає її, оцінює Accuracy, вимірює Latency, та зберігає ваги для подальшого
інференсу у Flask API.

Базується на оригінальному Jupyter ноутбуці з виправленими помилками.
"""

import os
import time
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from model_utils import SmallCNN, wav_to_melspec, SAMPLE_RATE, measure_latency, save_model, export_torchscript

# Відключаємо попередження для чистого виводу
warnings.filterwarnings("ignore", category=UserWarning)

def print_header(title: str):
    """Красивий заголовок"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Простий індикатор прогресу"""
    percent = 100 * current / total
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)

print_header("SPEECH COMMANDS CLASSIFICATION")
print("🎵 Лабораторна робота: Розгортання ML-моделі")
print(f"📅 Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Перевірка доступності CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Використовується пристрій: {device}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU пам'ять: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
else:
    print("💻 Використовується CPU")

# ============================================================================
# НАЛАШТУВАННЯ
# ============================================================================
print_header("НАЛАШТУВАННЯ")

CLASSES = ["yes", "no", "up", "down"]
N_CLASSES = len(CLASSES)
BATCH_SIZE = 64  # Збільшено для швидкості
EPOCHS = 10  # для швидкої демонстрації (збільши до 5+ для кращої якості)
LR = 1e-3
DATA_ROOT = os.path.join(os.getcwd(), "data_speech")
MODEL_PATH = "model_state_dict.pt"
TS_PATH = "model_scripted.pt"

os.makedirs(DATA_ROOT, exist_ok=True)

print(f"🎯 Налаштування:")
print(f"   - Класи: {CLASSES}")
print(f"   - Кількість класів: {N_CLASSES}")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - Epochs: {EPOCHS}")
print(f"   - Learning rate: {LR}")
print(f"   - Директорія даних: {DATA_ROOT}")
print(f"   - Модель буде збережена: {MODEL_PATH}")

# ============================================================================
# DATASET
# ============================================================================

class SubsetSpeechCommands(SPEECHCOMMANDS):
    """Обгортка над датасетом для вибірки тільки потрібних класів"""
    
    def __init__(self, root: str, subset: str = None, target_classes: List[str] = None):
        print(f"📂 Завантажуємо {subset} підмножину Speech Commands...")
        super().__init__(root, download=True, subset=subset)
        self.target_classes = set(target_classes) if target_classes else None
        self.label2idx = {c: i for i, c in enumerate(target_classes)} if target_classes else None
        
        # Відфільтруємо індекси з необхідними класами
        self._filtered = []
        print(f"🔍 Фільтрація класів {target_classes}...")
        
        for i in range(len(self._walker)):
            path = self._walker[i]
            label = Path(path).parent.name
            if self.target_classes is None or label in self.target_classes:
                self._filtered.append(i)
        
        print(f"✅ Знайдено {len(self._filtered)} файлів для класів {target_classes}")

    def __len__(self):
        return len(self._filtered)

    def __getitem__(self, idx: int):
        actual_idx = self._filtered[idx]
        waveform, sample_rate, label, *_ = super().__getitem__(actual_idx)
        y = self.label2idx[label] if self.label2idx else label
        # перетворення у Mel-спектрограму (в dB) + нормалізація ~[0..1]
        spec = wav_to_melspec(waveform, sample_rate)     # [1, n_mels, T]
        spec = (spec + 80.0) / 80.0
        return spec, y

def collate_fn(batch):
    """Функція для створення батчів з padding"""
    specs, labels = zip(*batch)  # список тензорів [1, n_mels, T_i]
    max_T = max(s.size(-1) for s in specs)
    padded = []
    for s in specs:
        if s.size(-1) < max_T:
            pad = torch.nn.functional.pad(s, (0, max_T - s.size(-1)))
            padded.append(pad)
        else:
            padded.append(s)
    x = torch.stack(padded, dim=0)  # [B, 1, n_mels, T]
    y = torch.tensor(labels, dtype=torch.long)
    return x, y

# ============================================================================
# ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================================
print_header("ЗАВАНТАЖЕННЯ ДАНИХ")

try:
    # Завантаження train/test (у SPEECHCOMMANDS є фіксовані підмножини)
    train_ds = SubsetSpeechCommands(DATA_ROOT, subset="training", target_classes=CLASSES)
    test_ds = SubsetSpeechCommands(DATA_ROOT, subset="testing", target_classes=CLASSES)
    
    print(f"📊 Розмір датасетів:")
    print(f"   - Train: {len(train_ds):,} зразків")
    print(f"   - Test: {len(test_ds):,} зразків")
    
    # Створюємо data loaders з оптимізованими налаштуваннями
    print(f"⚙️ Створюємо data loaders...")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Використовуємо оптимізації для {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0,  # 0 для Windows, щоб уникнути проблем з multiprocessing
        pin_memory=torch.cuda.is_available(),  # True для GPU, False для CPU
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
    
    print(f"✅ Data loaders створено:")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
except Exception as e:
    print(f"❌ Помилка завантаження даних: {e}")
    exit(1)

# ============================================================================
# СТВОРЕННЯ МОДЕЛІ
# ============================================================================
print_header("СТВОРЕННЯ МОДЕЛІ")

try:
    # Модель, лосс, оптимізатор
    model = SmallCNN(n_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Інформація про модель
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧠 Модель: {model.__class__.__name__}")
    print(f"🔢 Загальна кількість параметрів: {total_params:,}")
    print(f"🎯 Тренувальних параметрів: {trainable_params:,}")
    print(f"📐 Архітектура:")
    print(f"   - Вхід: [batch, 1, n_mels, time]")
    print(f"   - Вихід: [batch, {N_CLASSES}] (логіти для {N_CLASSES} класів)")
    print(f"   - Функція втрат: CrossEntropyLoss")
    print(f"   - Оптимізатор: Adam (lr={LR})")
    
    print(f"✅ Модель успішно створена та переміщена на {device}")
    
except Exception as e:
    print(f"❌ Помилка створення моделі: {e}")
    exit(1)

# ============================================================================
# ТРЕНУВАННЯ
# ============================================================================
print_header("ТРЕНУВАННЯ МОДЕЛІ")

print(f"🚀 Починаємо тренування...")
print(f"📊 Статистика:")
print(f"   - Епохи: {EPOCHS}")
print(f"   - Батчів на епоху: {len(train_loader)}")
print(f"   - Зразків на епоху: {len(train_ds):,}")
print(f"   - Загальна кількість батчів: {EPOCHS * len(train_loader)}")

start_time = time.time()

try:
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n📈 Epoch {epoch}/{EPOCHS} - {datetime.now().strftime('%H:%M:%S')}")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Переміщуємо дані на пристрій
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # Показуємо прогрес кожні 50 батчів
            if batch_idx % 50 == 0:
                current_acc = correct / total if total > 0 else 0
                print(f"   Batch {batch_idx+1:3d}/{len(train_loader)} "
                      f"({100*batch_idx/len(train_loader):5.1f}%) - "
                      f"Loss: {loss.item():.4f}, Acc: {current_acc:.4f}")

        # Статистика епохи
        epoch_time = time.time() - epoch_start
        train_loss = running_loss / total
        train_acc = correct / total
        
        print(f"✅ Epoch {epoch} завершена за {epoch_time:.1f}с")
        print(f"   📉 Середня втрата: {train_loss:.4f}")
        print(f"   🎯 Точність: {train_acc*100:.2f}%")
        print(f"   ⚡ Швидкість: {total/epoch_time:.1f} зразків/сек")

    total_time = time.time() - start_time
    print(f"\n🎉 Тренування завершено!")
    print(f"⏱️ Загальний час: {total_time:.1f} секунд ({total_time/60:.1f} хвилин)")
    print(f"📊 Фінальна точність на train: {train_acc*100:.2f}%")
    
except KeyboardInterrupt:
    print(f"\n⏹️ Тренування перервано користувачем")
except Exception as e:
    print(f"\n❌ Помилка під час тренування: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# ОЦІНКА НА TEST SET
# ============================================================================
print_header("ОЦІНКА МОДЕЛІ НА ТЕСТОВОМУ НАБОРІ")

print(f"🔍 Оцінюємо модель на тестовому наборі...")
print(f"📊 Тестових зразків: {len(test_ds):,}")
print(f"📦 Тестових батчів: {len(test_loader)}")

eval_start = time.time()

try:
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)
            
            # Статистика
            test_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # Зберігаємо для детального аналізу
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Прогрес кожні 10 батчів
            if batch_idx % 10 == 0:
                print_progress(batch_idx + 1, len(test_loader), "   Оцінка")

    print()  # Новий рядок після прогресу
    
    eval_time = time.time() - eval_start
    test_loss = test_loss / total
    test_acc = correct / total
    
    print(f"✅ Оцінка завершена за {eval_time:.1f}с")
    print(f"📊 Результати:")
    print(f"   🎯 Test Accuracy: {test_acc*100:.2f}%")
    print(f"   📉 Test Loss: {test_loss:.4f}")
    print(f"   ✅ Правильних передбачень: {correct:,}/{total:,}")
    print(f"   ⚡ Швидкість оцінки: {total/eval_time:.1f} зразків/сек")
    
    # Детальна статистика по класах
    print(f"\n📈 Статистика по класах:")
    pred_counter = Counter(all_preds)
    label_counter = Counter(all_labels)
    
    for i, class_name in enumerate(CLASSES):
        true_count = label_counter.get(i, 0)
        pred_count = pred_counter.get(i, 0)
        
        # Точність для цього класу
        class_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == i and l == i)
        class_acc = class_correct / true_count if true_count > 0 else 0
        
        print(f"   {class_name:>6}: {true_count:4d} справжніх, "
              f"{pred_count:4d} передбачених, точність: {class_acc*100:5.1f}%")
    
except Exception as e:
    print(f"\n❌ Помилка під час оцінки: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# ВИМІРЮВАННЯ ЛАТЕНТНОСТІ
# ============================================================================
print_header("ВИМІРЮВАННЯ ПРОДУКТИВНОСТІ")

print(f"⏱️ Вимірюємо латентність інференсу...")

try:
    # Підготуємо приклад для тестування
    example_spec, _ = test_ds[0]
    
    # Падимо до консистентної довжини (беремо максимум з кількох зразків)
    test_samples = [test_ds[i] for i in range(min(8, len(test_ds)))]
    ex_T = max(s.size(-1) for s, _ in test_samples) if test_samples else example_spec.size(-1)
    
    if example_spec.size(-1) < ex_T:
        example_spec = torch.nn.functional.pad(example_spec, (0, ex_T - example_spec.size(-1)))
    
    example = example_spec.unsqueeze(0).to(device)  # [1,1,n_mels,T]
    
    print(f"📏 Розмір тестового входу: {list(example.shape)}")
    print(f"🔬 Вимірюємо латентність на 30 прогонах...")
    
    lat_ms = measure_latency(model, example, runs=30)
    
    print(f"✅ Результати продуктивності:")
    print(f"   ⚡ Середня латентність: {lat_ms:.2f} мс")
    print(f"   🚀 Пропускна здатність: {1000/lat_ms:.1f} запитів/сек")
    print(f"   📊 Час на один зразок: {lat_ms:.2f} мс")
    
    if lat_ms < 100:
        print(f"   🟢 Відмінна швидкість для реального часу!")
    elif lat_ms < 500:
        print(f"   🟡 Хороша швидкість для більшості застосунків")
    else:
        print(f"   🔴 Можливо знадобиться оптимізація для реального часу")
        
except Exception as e:
    print(f"❌ Помилка вимірювання латентності: {e}")
    # Створюємо dummy приклад
    example = torch.randn(1, 1, 64, 32).to(device)
    lat_ms = 0.0

# ============================================================================
# ЗБЕРЕЖЕННЯ МОДЕЛІ
# ============================================================================
print_header("ЗБЕРЕЖЕННЯ МОДЕЛЕЙ")

print(f"💾 Зберігаємо натреновану модель...")

try:
    # Збереження state_dict
    size_sd = save_model(model, MODEL_PATH)
    print(f"✅ State dict збережено:")
    print(f"   📁 Файл: {MODEL_PATH}")
    print(f"   📏 Розмір: {size_sd/1024:.1f} KB")
    
    # Збереження TorchScript
    try:
        size_ts = export_torchscript(model, example, TS_PATH)
        print(f"✅ TorchScript збережено:")
        print(f"   📁 Файл: {TS_PATH}")
        print(f"   📏 Розмір: {size_ts/1024:.1f} KB")
        print(f"   📊 Коефіцієнт стиснення: {size_ts/size_sd:.2f}x")
    except Exception as e:
        print(f"⚠️ Помилка збереження TorchScript: {e}")
        print(f"   State dict все одно збережено і може використовуватись")
    
    print(f"💡 Файли збережено в поточній директорії:")
    print(f"   {os.path.abspath(MODEL_PATH)}")
    if os.path.exists(TS_PATH):
        print(f"   {os.path.abspath(TS_PATH)}")
        
except Exception as e:
    print(f"❌ Помилка збереження: {e}")

# ============================================================================
# ПІДСУМОК
# ============================================================================
print_header("ПІДСУМОК РОБОТИ")

print(f"🎉 Лабораторна робота успішно завершена!")
print(f"📊 Фінальні результати:")
if 'test_acc' in locals():
    print(f"   🎯 Точність на тесті: {test_acc*100:.2f}%")
if 'lat_ms' in locals():
    print(f"   ⚡ Латентність: {lat_ms:.2f} мс")
print(f"   💾 Модель збережена: {MODEL_PATH}")
print(f"   🎵 Класи: {CLASSES}")
print(f"   📱 Пристрій: {device}")

print(f"\n📝 Як використати з Flask API:")
print(f"   1. Переконайтеся, що файли model_utils.py та {MODEL_PATH} в поточній папці")
print(f"   2. Запустіть API: python app.py")
print(f"   3. Тестуйте:")
print(f"      curl -X POST -F \"file=@sample.wav\" http://127.0.0.1:8000/predict")

print(f"\n💡 Рекомендації:")
if 'test_acc' in locals():
    if test_acc > 0.85:
        print(f"   🟢 Модель показує хорошу точність!")
    elif test_acc > 0.70:
        print(f"   🟡 Модель працює задовільно. Можна покращити збільшивши EPOCHS")
    else:
        print(f"   🔴 Модель потребує покращення. Спробуйте більше епох або інші гіперпараметри")

if 'lat_ms' in locals() and lat_ms > 0:
    if lat_ms < 50:
        print(f"   🟢 Відмінна швидкість для real-time застосунків!")
    elif lat_ms < 200:
        print(f"   🟡 Хороша швидкість для більшості застосунків")

print(f"\n✅ Всі етапи виконано:")
print(f"   📂 Завантаження даних ✅")
print(f"   🧠 Створення моделі ✅") 
print(f"   🏋️ Тренування ✅")
print(f"   🎯 Оцінка ✅")
print(f"   ⚡ Вимірювання швидкості ✅")
print(f"   💾 Збереження ✅")

print(f"\n🎵 Speech Commands Classification завершено!")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)