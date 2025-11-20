# STEM Math Solving with Vision-Language Models

Решение для задачи автоматического решения математических и физических задач с изображениями, используя мультимодальные модели машинного обучения

## Описание проекта

Проект предназначен для анализа геометрических и физических задач, представленных в виде изображения и текстового вопроса. Система определяет правильность утверждений на основе анализа визуальной информации и текста.

<img width="512" height="632" alt="image" src="https://github.com/user-attachments/assets/deecbc69-3e0e-4a3a-a330-87097fb91427" />


```
A. AB = CD B. ∠AEB = 30° C. AB = 4 cm D. BC = 4 cm E. BE ⊥ BC

Ответ: "AC"
```

**Формат ввода**

Поаётся файл input.pickle. Он содержит список со словарями, в которых содержатся картинки и варианты ответов:
```
[
    {
        "rid" : 1,
        "question" : "A. Angle α is 86°\nB. The angle adjacent to angle α is 56°\nC. α = 48°\nD. The sum of angles 86° and 38° is equal to angle α\nE. All the rays depicted lie on two intersecting lines",
        "image" : <blob>,
    },
    {
        "rid" : 2,
        "question" : "What can be said about the liquid levels h₁ and h₂ shown in the figure?\nA. h₁ > h₂\nB. h₂ > h₁\nC. h₁ = h₂\nD. This cannot be determined from the figure.",
        "image" : <blob>,
    }
]
```

**Формат вывода**
solution.py должен записывает output.json со списком словарей с ответами, например:

```
[{"rid": 1, "answer": "BC"}, {"rid": 2, "answer": "D"}]
```
## Tech Stack

**Computer Vision & Deep Learning:**
- PyTorch (нейронные сети, обучение моделей)
- Transformers (Hugging Face): Qwen2-VL-7B-Instruct, InternVL2-8B
- OpenCV (обработка и анализ изображений)
- Pillow (работа с изображениями)
- 4-bit квантизация моделей (BitsAndBytes)

**Machine Learning:**
- LightGBM / XGBoost (градиентный бустинг для мета-модели)
- scikit-learn (классификация, метрики)
- NumPy (векторные вычисления)

**NLP:**
- Natural Language Inference модели
- Few-shot learning с промптами
- Self-consistency для повышения надежности

## Архитектура

**Многоуровневый ансамбль экспертных моделей:**

### 1. Vision-Language Models (vlm.py)
- Qwen2-VL-7B-Instruct, InternVL2-8B
- 4-bit квантизация для работы на ограниченных ресурсах
- Self-consistency: множественная генерация с агрегацией
- Few-shot промптинг с примерами задач
- Анализ энтропии и консистентности предсказаний

### 2. OCR + Rule-based система (ocr_rules.py)
- Извлечение текста и численных данных с изображений
- Feature engineering: признаки изображения (размеры, aspect ratio, цвета)
- Feature engineering: признаки вопроса (тип задачи, ключевые слова)
- Специализированные эвристики для разных типов задач

### 3. Natural Language Inference (nli.py)
- Логический анализ текстовых утверждений
- Детекция противоречий и следствий
- Вероятностная оценка (entailment/contradiction/neutral)

### 4. Мета-модель (meta.py)
- LightGBM/XGBoost для финального решения
- Взвешенное голосование с учетом уверенности экспертов
- Разрешение конфликтов между моделями
- Калибровка вероятностей и оптимизация порогов

## Структура проекта

```
1114-solution/
├── solution.py          # Основное решение с эвристиками
├── vlm.py              # VLM модели (Qwen2-VL, InternVL2)
├── ocr_rules.py        # OCR + правила для геометрии
├── nli.py              # Natural Language Inference
├── meta.py             # Мета-модель для объединения результатов
├── synthetic_data.py   # Генерация синтетических данных
├── benchmark_local.py  # Локальное тестирование
├── quick_test.py       # Быстрые тесты
├── requirements.txt    # Зависимости
└── Dockerfile         # Docker-образ для запуска
```

## Установка и запуск

### Требования
- Python 3.8+
- CUDA-совместимый GPU (опционально, для VLM моделей)
- 8+ GB RAM

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск решения
python solution.py

# Тестирование
python benchmark_local.py
```

Входные данные: `input.pickle` (список объектов с полями `rid`, `question`, `image`)
Выходные данные: `output.json` (предсказания в формате `{"rid": "...", "answer": "A"/"BC"/...}`)

## Ключевые технические решения

### Computer Vision
- **Мультимодальный анализ**: обработка изображений через Vision-Language Models
- **Feature extraction**: извлечение признаков изображения (размеры, цвета, aspect ratio)
- **OCR**: распознавание текста и чисел на схемах
- **Image preprocessing**: нормализация и адаптация под разные модели

### Machine Learning Pipeline
```
Изображение + Вопрос
        ↓
┌───────┴────────┬────────────┬──────────┐
│   VLM          │    OCR     │   NLI    │
│ (GPU/4-bit)    │  +Rules    │  Model   │
└───────┬────────┴────────────┴──────────┘
        ↓
    Мета-модель (LightGBM)
        ↓
   Финальный ответ
```

### Оптимизации
- **4-bit квантизация**: уменьшение memory footprint моделей в 4 раза
- **Self-consistency**: генерация нескольких ответов с агрегацией для повышения надежности
- **Ensemble learning**: комбинация предсказаний разных архитектур
- **Адаптивные пороги**: динамическая калибровка уверенности для разных типов задач

## Feature Engineering

**Признаки изображения:**
- Размеры (width, height, area)
- Aspect ratio
- Цветовые характеристики (mean, variance)
- Морфологические признаки

**Признаки текста:**
- Длина и количество слов
- Наличие чисел и специальных символов
- Ключевые слова (angle, circle, perpendicular, triangle)
- Автоматическое определение типа задачи

**Мета-признаки:**
- Энтропия предсказаний
- Согласованность между экспертами
- Уверенность моделей
- Детекция конфликтов

## Метрики

- Accuracy (точность классификации)
- Entropy (неопределенность предсказаний)
- Consistency (согласованность между прогонами)
- Agreement score (согласованность экспертов)
- Calibration (калибровка вероятностей)

## Зависимости

```
torch>=1.9.0
transformers>=4.20.0
Pillow>=8.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
lightgbm (optional)
xgboost (optional)
```
