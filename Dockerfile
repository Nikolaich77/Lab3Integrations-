# ============================================
# STAGE 1: ТРЕНУВАННЯ МОДЕЛІ (GPU)
# ===========================================
# Використовуємо NVIDIA CUDA base image для GPU тренування
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS trainer

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Встановлення Python 3.11 та системних залежностей
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libsndfile1 \
    sox \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Копіювання requirements.txt та встановлення GPU версій залежностей
COPY requirements.txt .
# Встановлюємо CUDA версії torch та torchaudio (cu118 для CUDA 11.8)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir flask==2.2.5 numpy==1.25.2 requests==2.31.0 soundfile==0.12.1

# Копіювання ТІЛЬКИ необхідних файлів для тренування (БЕЗ model.pth!)
# Копіюємо потрібні файли для тренування
COPY speech_commands_train.py model_utils.py requirements.txt ./

# ЗА ЗАМОВЧУВАННЯМ тренування виконується при збірці (можна відключити через --build-arg TRAIN_MODEL=false)
ARG TRAIN_MODEL=true

# Тренування моделі: виконується якщо TRAIN_MODEL=true (за замовчуванням)
RUN if [ "$TRAIN_MODEL" = "true" ]; then \
        echo "🏋️ Починаємо тренування моделі в Docker..." && \
        echo "📥 Завантаження датасету (може зайняти багато часу)..." && \
        python -u speech_commands_train.py && \
        echo "✅ Тренування завершено!" && \
        ls -lh model_state_dict.pt model_scripted.pt || true && \
        echo "📊 Моделі створено:" && \
        du -h model_state_dict.pt model_scripted.pt 2>/dev/null || true && \
        rm -rf /app/data_speech /app/SpeechCommands || true && \
        echo "🗑 Датасет видалено для зменшення розміру образу"; \
    else \
        echo "⏭ Пропуск тренування (TRAIN_MODEL=false). Моделі будуть тренуватись автоматично при першому запуску API."; \
    fi
# Перевірка наявності збережених моделей, якщо їх немає - створюємо порожні файли
RUN if [ ! -f model_state_dict.pt ]; then echo "" > model_state_dict.pt; fi
RUN if [ ! -f model_scripted.pt ]; then echo "" > model_scripted.pt; fi

# ============================================
# STAGE 2: ІНФЕРЕНС (PRODUCTION)
# ============================================
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Створення непривілейованого користувача
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Встановлення runtime залежностей
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    sox \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /root/.wget-hsts

WORKDIR /app 

# Встановлення мінімальних Python залежностей
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    pip install --no-cache-dir soundfile==0.12.1


COPY --from=trainer --chown=appuser:appuser /app/model_state_dict.pt ./model_state_dict.pt
COPY --from=trainer --chown=appuser:appuser /app/model_scripted.pt ./model_scripted.pt
COPY --chown=appuser:appuser app.py model_utils.py ./
COPY --chown=appuser:appuser templates/ ./templates/

# Створення директорій
RUN mkdir -p /app/uploads /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Контейнер слухає порт 8000 (app.py за замовчуванням використовує 8000)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["python", "app.py"]