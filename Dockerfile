# Комплексное решение с VLM, OCR, NLI и мета-моделью
FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
RUN pip3 install --no-cache-dir \
    pandas \
    pillow \
    opencv-python \
    numpy \
    scipy \
    scikit-learn \
    transformers \
    torch \
    torchvision \
    accelerate \
    bitsandbytes \
    paddlepaddle \
    paddleocr \
    lightgbm \
    xgboost \
    matplotlib \
    seaborn \
    tqdm

# Установка дополнительных зависимостей для VLM
RUN pip3 install --no-cache-dir \
    sentence-transformers \
    datasets \
    evaluate \
    rouge-score \
    nltk

WORKDIR /workspace
COPY . .

# Создание директории для кэша моделей
RUN mkdir -p /workspace/.cache

# Установка переменных окружения
ENV TRANSFORMERS_CACHE=/workspace/.cache
ENV HF_HOME=/workspace/.cache
ENV CUDA_VISIBLE_DEVICES=0

# Комплексное решение с ансамблем экспертов
ENTRYPOINT ["python3", "solution.py"]
