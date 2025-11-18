#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API для Speech Commands Classification

Цей скрипт створює REST API для класифікації аудіо команд 
використовуючи навчену CNN модель.
"""

import os
import io
import warnings
from typing import Dict, Any

import torch
import torchaudio
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import time

from model_utils import SmallCNN, wav_to_melspec, load_model

# Відключаємо попередження
warnings.filterwarnings("ignore", category=UserWarning)

# Налаштування
CLASSES = ["yes", "no", "up", "down"]
N_CLASSES = len(CLASSES)
MODEL_PATH = "model_state_dict.pt"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Глобальні змінні
model = None
device = None


def check_model_validity(path: str) -> bool:
    """Перевірка чи файл моделі валідний"""
    if not os.path.exists(path):
        return False
    
    # Перевіряємо розмір (placeholder файли дуже малі)
    if os.path.getsize(path) < 1000:  # менше 1KB = placeholder
        return False
    
    # Пробуємо завантажити
    try:
        torch.load(path, map_location='cpu', weights_only=True)
        return True
    except Exception:
        # Пробуємо без weights_only
        try:
            torch.load(path, map_location='cpu')
            return True
        except Exception:
            return False


def train_model_if_needed():
    """Тренує модель якщо файли відсутні або невалідні"""
    need_training = False
    
    print("\n🔍 Перевірка наявності моделей...")
    
    if not check_model_validity(MODEL_PATH):
        print(f"⚠️ Файл {MODEL_PATH} відсутній або невалідний")
        need_training = True
    else:
        print(f"✅ {MODEL_PATH} валідний")
    
    scripted_path = "model_scripted.pt"
    if not check_model_validity(scripted_path):
        print(f"⚠️ Файл {scripted_path} відсутній або невалідний")
    else:
        print(f"✅ {scripted_path} валідний")
    
    if need_training:
        print("\n" + "="*60)
        print("🏋️ АВТОМАТИЧНЕ ТРЕНУВАННЯ МОДЕЛІ")
        print("="*60)
        print("⚠️ Моделі не знайдено, починаємо тренування...")
        print("⏱️ Це може зайняти 5-60 хвилин залежно від пристрою\n")
        
        try:
            # Імпортуємо та запускаємо тренування
            import subprocess
            import sys
            
            result = subprocess.run(
                [sys.executable, "speech_commands_train.py"],
                cwd=os.getcwd(),
                capture_output=False,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Тренування завершилось з помилкою (код {result.returncode})")
            
            print("\n✅ Автоматичне тренування завершено!")
            
            # Перевіряємо що файл створено
            if not check_model_validity(MODEL_PATH):
                raise FileNotFoundError(f"Після тренування {MODEL_PATH} все ще відсутній")
                
        except Exception as e:
            print(f"\n❌ Помилка автоматичного тренування: {e}")
            raise RuntimeError(
                f"Не вдалося автоматично натренувати модель. "
                f"Будь ласка, запустіть вручну: python speech_commands_train.py"
            )


def init_model():
    """Ініціалізація моделі"""
    global model, device
    
    print("🚀 Ініціалізація Speech Commands API...")
    
    # Пристрій
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Використовується пристрій: {device}")
    
    # Перевіряємо та тренуємо модель якщо потрібно
    train_model_if_needed()
    
    # Завантаження моделі
    try:
        model = load_model_safe(SmallCNN, MODEL_PATH, N_CLASSES, device)
        print(f"✅ Модель завантажена: {MODEL_PATH}")
        print(f"🔢 Кількість класів: {N_CLASSES}")
        print(f"📂 Класи: {CLASSES}")
        
        # Тест моделі
        test_input = torch.randn(1, 1, 64, 32).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            print(f"🧪 Тест моделі пройдено: вихід розміру {test_output.shape}")
            
    except Exception as e:
        raise RuntimeError(f"❌ Помилка завантаження моделі: {e}")


def create_directories():
    """Створення необхідних директорій"""
    directories = ['templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Створено директорію: {directory}")


def load_model_safe(model_class, path: str, n_classes: int = 4, device_param: torch.device = None) -> torch.nn.Module:
    """Безпечне завантаження моделі з обробкою помилок"""
    if device_param is None:
        device_param = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class(n_classes=n_classes)
    
    try:
        # Завантажуємо state_dict
        state_dict = torch.load(path, map_location=device_param, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        # Якщо не вдалося завантажити з weights_only=True, пробуємо без нього
        print(f"⚠️ Не вдалося завантажити з weights_only=True, пробуємо інший спосіб...")
        state_dict = torch.load(path, map_location=device_param)
        model.load_state_dict(state_dict)
    
    model.to(device_param)
    model.eval()
    return model


def is_allowed_file(filename: str) -> bool:
    """Перевірка дозволених форматів файлів"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    """Предобробка аудіо файлу"""
    import tempfile
    import os as os_module
    
    temp_path = None
    try:
        # Створюємо тимчасовий файл для torchaudio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = tmp_file.name
        
        # Завантажуємо аудіо з тимчасового файлу
        waveform, sample_rate = torchaudio.load(temp_path)
        
        # Конвертуємо в моно, якщо стерео
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Обмежуємо довжину (максимум 1 секунда)
        max_samples = 16000  # 1 секунда при 16kHz
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        
        # Перетворюємо в Mel-спектрограму
        spec = wav_to_melspec(waveform, sample_rate)
        spec = (spec + 80.0) / 80.0  # Нормалізація
        
        return spec.unsqueeze(0)  # [1, 1, n_mels, T]
        
    except Exception as e:
        raise ValueError(f"Помилка обробки аудіо: {e}")
    finally:
        # Видаляємо тимчасовий файл
        if temp_path and os_module.path.exists(temp_path):
            try:
                os_module.unlink(temp_path)
            except:
                pass


def predict_audio(spec: torch.Tensor) -> Dict[str, Any]:
    """Передбачення для аудіо спектрограми"""
    try:
        spec = spec.to(device)
        
        with torch.no_grad():
            logits = model(spec)
            probs = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            
        # Формуємо результат
        predicted_class = CLASSES[predicted_idx]
        probabilities = {
            CLASSES[i]: float(probs[0][i].item()) 
            for i in range(N_CLASSES)
        }
        
        return {
            "predicted": predicted_class,
            "probabilities": probabilities,
            "confidence": float(probs[0][predicted_idx].item())
        }
        
    except Exception as e:
        raise RuntimeError(f"Помилка передбачення: {e}")


@app.route('/')
def index():
    """Головна сторінка з веб-інтерфейсом"""
    return render_template('index.html')


@app.route('/api/', methods=['GET'])
def api_home():
    """API інформація"""
    return jsonify({
        "name": "Speech Commands Classification API",
        "version": "1.0.0",
        "description": "API для класифікації аудіо команд (yes, no, up, down)",
        "endpoints": {
            "GET /": "Веб-інтерфейс",
            "GET /api/": "API інформація",
            "POST /predict": "Класифікація аудіо файлу",
            "GET /health": "Перевірка статусу сервісу",
            "GET /info": "Інформація про модель"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": f"{MAX_FILE_SIZE // (1024*1024)} MB",
        "classes": CLASSES
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Перевірка здоров'я сервісу"""
    try:
        # Простий тест моделі
        dummy_input = torch.randn(1, 1, 64, 32).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        return jsonify({
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device)
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/info', methods=['GET'])
def model_info():
    """Інформація про модель"""
    return jsonify({
        "model": "SmallCNN",
        "classes": CLASSES,
        "n_classes": N_CLASSES,
        "device": str(device),
        "model_file": MODEL_PATH,
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Основний endpoint для класифікації"""
    try:
        # Перевірка чи модель завантажена
        if model is None:
            print("❌ ПОМИЛКА: Модель не завантажена!")
            return jsonify({"error": "Модель не завантажена. Перезапустіть сервер."}), 503
        
        # Перевірка наявності файлу
        if 'file' not in request.files:
            print("❌ Файл не знайдено в запиті")
            return jsonify({"error": "Файл не знайдено в запиті"}), 400
        
        file = request.files['file']
        
        # Перевірка імені файлу
        if file.filename == '':
            print("❌ Файл не вибрано")
            return jsonify({"error": "Файл не вибрано"}), 400
        
        print(f"📁 Отримано файл: {file.filename}")
        
        # Перевірка формату
        if not is_allowed_file(file.filename):
            print(f"❌ Непідтримуваний формат: {file.filename}")
            return jsonify({
                "error": f"Непідтримуваний формат файлу. Дозволені: {list(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Читаємо файл
        audio_bytes = file.read()
        print(f"📊 Розмір файлу: {len(audio_bytes)} байт")
        
        # Перевіряємо розмір
        if len(audio_bytes) == 0:
            print("❌ Порожній файл")
            return jsonify({"error": "Порожній файл"}), 400
        
        # Предобробка аудіо
        try:
            print("🔄 Починаємо обробку аудіо...")
            spec = preprocess_audio(audio_bytes)
            print(f"✅ Аудіо оброблено, розмір спектрограми: {spec.shape}")
        except Exception as e:
            print(f"❌ Помилка обробки аудіо: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Помилка обробки аудіо: {str(e)}"}), 400
        
        # Передбачення
        try:
            print("🤖 Починаємо передбачення...")
            result = predict_audio(spec)
            print(f"✅ Передбачено: {result['predicted']} (впевненість: {result['confidence']:.2%})")
            
            # Додаємо метадані
            result.update({
                "filename": file.filename,
                "model": "SmallCNN",
                "classes": CLASSES,
                "timestamp": time.time()
            })
            
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Помилка передбачення: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Помилка передбачення: {str(e)}"}), 500
        
    except Exception as e:
        print(f"❌ Внутрішня помилка: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Внутрішня помилка сервера: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(e):
    """Обробка занадто великих файлів"""
    return jsonify({
        "error": f"Файл занадто великий. Максимальний розмір: {MAX_FILE_SIZE // (1024*1024)} MB"
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Обробка 404"""
    return jsonify({
        "error": "Endpoint не знайдено",
        "available_endpoints": ["/", "/health", "/info", "/predict"]
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Обробка внутрішніх помилок"""
    return jsonify({
        "error": "Внутрішня помилка сервера",
        "message": "Перевірте логи сервера для деталей"
    }), 500


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("🎵 Speech Commands Classification API")
        print("=" * 60)
        
        # Створюємо необхідні директорії
        create_directories()
        
        # Ініціалізація моделі
        init_model()

        print("\n🌟 Speech Commands API з веб-інтерфейсом запущено!")
        print("📋 Доступні endpoints:")
        print("   GET  /          - Веб-інтерфейс")
        print("   GET  /api/      - API інформація")
        print("   GET  /health    - Перевірка статусу")
        print("   GET  /info      - Інформація про модель")
        print("   POST /predict   - Класифікація аудіо")

        print("\n🌐 Відкрийте у браузері (якщо доступно):")
        host_print = os.environ.get('HOST', '0.0.0.0')
        port_print = os.environ.get('PORT', '8000')
        print(f"   http://{host_print}:{port_print}/")

        print("\n💡 Приклад API запиту:")
        print("   curl -X POST -F \"file=@your_audio.wav\" http://127.0.0.1:8000/predict")

        print("\n🔧 Налаштування:")
        print(f"   - Підтримувані формати: {list(ALLOWED_EXTENSIONS)}")
        print(f"   - Максимальний розмір файлу: {MAX_FILE_SIZE // (1024*1024)} MB")
        print(f"   - Класи: {CLASSES}")
        print(f"   - Модель: {MODEL_PATH}")
        print(f"   - Пристрій: {device}")

        print("\n" + "=" * 60)
        print("🚀 Сервер запускається...")
        print("📝 Для зупинки натисніть Ctrl+C")
        print("=" * 60)
        
        # Запуск Flask app
        # Підтримуємо конфігурацію хоста/порту через змінні оточення (зручніше для Docker)
        run_host = os.environ.get('HOST', '0.0.0.0')
        run_port = int(os.environ.get('PORT', 8000))
        app.run(
            host=run_host,
            port=run_port,
            debug=False,  # False для продакшену
            threaded=True,
            use_reloader=False  # Відключаємо reloader щоб не було подвійної ініціалізації
        )
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Сервер зупинено користувачем")
        print("👋 До побачення!")
        
    except Exception as e:
        print(f"\n❌ Критична помилка запуску API: {e}")
        print("\n🔍 Перевірте:")
        print("   1. Чи існує файл model_state_dict.pt?")
        print("   2. Чи встановлені всі залежності?")
        print("   3. Чи створена папка templates/ з файлом index.html?")
        print("\n📖 Запустіть спочатку: python speech_commands_train.py")
        
        import traceback
        print(f"\n🐛 Детальна помилка:")
        traceback.print_exc()