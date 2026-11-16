# 📘 Практическое руководство: Извлечение признаков (Feature Extraction) для локального ML

Руководство по выбору и запуску **компактных, актуальных моделей (весом от 10 до 150 МБ)** для извлечения плотных векторных представлений (эмбеддингов) из любых типов данных на обычном процессоре (CPU) или слабом GPU.

---

## 🧭 Содержание
1. [Почему извлечение признаков (Feature Extraction) побеждает Fine-Tuning?](#1-почему-feature-extraction-побеждает-fine-tuning)
2. [Обзор компактных моделей по 6 типам данных](#2-обзор-компактных-моделей-по-6-типам-данных)
   - [2.1 Текстовые данные (NLP)](#21-текстовые-данные-nlp)
   - [2.2 Изображения (Computer Vision)](#22-изображения-computer-vision)
   - [2.3 Мультимодальные данные (Image + Text)](#23-мультимодальные-данные-image--text)
   - [2.4 Аудио и Речь (Audio & Speech)](#24-аудио-и-речь-audio--speech)
   - [2.5 Временные ряды (Time Series)](#25-временные-ряды-time-series)
   - [2.6 Табличные данные (Tabular Data)](#26-табличные-данные-tabular-data)
3. [Продвинутые техники оптимизации (Matryoshka, ONNX, INT8)](#3-продвинутые-техники-оптимизации)
4. [Сводная матрица выбора инженера](#4-сводная-матрица-выбора-инженера)

---

## 1. Почему Feature Extraction побеждает Fine-Tuning?

При решении прикладных задач инженеры часто совершают ошибку, пытаясь дообучать (fine-tune) тяжелые нейросети целиком.

| Параметр | Полное дообучение (Fine-Tuning) | Замороженный экстрактор + GBDT / Линейная голова |
|:---|:---:|:---:|
| **Требования к железу** | Мощный GPU с 16–24 ГБ VRAM | Обычный процессор (CPU) или ноутбук |
| **Время обучения** | Часы / дни | **1–5 секунд** (обучение LightGBM на эмбеддингах) |
| **Объем датасета** | Требуются десятки тысяч примеров | Работает даже на **50–200 примерах** без переобучения |
| **Мультимодальность** | Крайне сложная архитектура сшивки градиентов | Простое объединение векторов (`np.hstack([text_emb, img_emb, table])`) |

---

## 2. Обзор компактных моделей по 6 типам данных

---

### 2.1 Текстовые данные (NLP)

#### Топ моделей для локального запуска:
1. **`all-MiniLM-L6-v2` (~80 МБ, $d=384$):**
   * Мировой эталон для легких задач семантического поиска, кластеризации и классификации.
   * Скорость: ~5–10 мс на CPU.
2. **`multilingual-e5-small` (~140 МБ, $d=384$):**
   * Превосходная поддержка русского языка и еще 100+ языков. Для запросов поиска используется префикс `query: `, для документов — `passage: `.
3. **`nomic-embed-text-v1.5` (~130 МБ, $d=64..768$):**
   * Поддержка сверхдлинного контекста до **8192 токенов** и технологии Matryoshka (можно отрезать первые 128 чисел без потери качества).

#### Рабочий код:
```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

texts = ["Система обнаружения дефектов", "Банковский перевод за границу"]
embeddings = model.encode(texts, normalize_embeddings=True)  # Shape: (2, 384)
```

---

### 2.2 Изображения (Computer Vision)

#### Топ моделей для локального запуска:
1. **`MobileNetV3-Small / MobileNetV4` (~9–15 МБ, $d=576..960$):**
   * Минимальное потребление памяти, молниеносный инференс на мобильных чипах и слабом CPU.
2. **`DINOv2-Small (vits14)` (~85 МБ, $d=384$):**
   * Разработка Meta AI. **Абсолютный лидер по качеству эмбеддингов без учителя**. Находит мелкие детали, дефекты и контуры без необходимости файнтюнинга.
3. **`EfficientNet-B0` (~20 МБ, $d=1280$):**
   * Сбалансированная сверточная сеть с глубоким пониманием объектов ImageNet.

#### Рабочий код:
```python
# pip install torch torchvision pillow
import torch
import torchvision.models as models
from PIL import Image

weights = models.MobileNet_V3_Small_Weights.DEFAULT
backbone = models.mobilenet_v3_small(weights=weights)
backbone.classifier = torch.nn.Identity()  # Извлекаем предпоследний слой
backbone.eval()

transform = weights.transforms()
img = Image.open("sample.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    features = backbone(tensor).numpy().flatten()  # Shape: (576,)
```

---

### 2.3 Мультимодальные данные (Image + Text)

Позволяют помещать изображения и текст в единое геометрическое пространство (поиск картинок по смысловому описанию, Zero-Shot классификация).

#### Топ моделей:
1. **`MobileCLIP-S0` (Apple, ~35 МБ, $d=512$):**
   * Спроектирована специально для мгновенного инференса на потребительских устройствах. Превосходит классический OpenAI CLIP-B/32 при вчетверо меньшем размере.
2. **`SigLIP-base-patch16` (Google, ~180 МБ, $d=768$):**
   * Преемник CLIP на базе сигмоидной функции потерь. Значительно точнее ранжирует пары «текст-картинка».

#### Рабочий код:
```python
# pip install transformers torch pillow
from transformers import AutoProcessor, AutoModel
from PIL import Image

model_id = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

img = Image.open("car.jpg")
inputs = processor(text=["красный седан", "велосипед"], images=img, padding=True, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)
    img_vec = out.image_embeds   # (1, 768)
    text_vec = out.text_embeds  # (2, 768)
```

---

### 2.4 Аудио и Речь (Audio & Speech)

#### Топ моделей:
1. **`Whisper-tiny (Encoder)` (OpenAI, ~75 МБ, $d=384$):**
   * Извлечение скрытых состояний энкодера речи. Позволяет классифицировать язык, интонации, эмоции и голосовые команды.
2. **`CLAP` (LAION, ~120 МБ, $d=512$):**
   * Мультимодальный экстрактор для любых звуков (музыка, звон стекла, лай собак, индустриальный шум станков).
3. **`Wav2Vec 2.0 / HuBERT-base` (~95 МБ квантованная):**
   * Фонетические представления человеческой речи, биометрия диктора (Speaker ID).

#### Рабочий код:
```python
# pip install transformers torch torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor
import torchaudio

model = WhisperModel.from_pretrained("openai/whisper-tiny")
extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

waveform, sr = torchaudio.load("speech.wav")
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    hidden = model.encoder(inputs.input_features).last_hidden_state
    speech_embedding = hidden.mean(dim=1).numpy().flatten()  # (384,)
```

---

### 2.5 Временные ряды (Time Series)

В анализе сенсоров и рядов тяжелые трансформеры часто уступают сверточным методам по скорости в 1000 раз при равной точности.

#### Топ методов:
1. **`MiniRocket` (aeon / sktime):**
   * 84 случайные недетерминированные свертки. Вычисляет признаки за миллисекунды на CPU без градиентного спуска.
2. **`Catch22` (C-библиотека):**
   * 22 фундаментальных статистических признака (автокорреляция, фрактальная размерность, энтропия распределения).
3. **`Chronos-Tiny` (Amazon, ~30 МБ, 8M параметров):**
   * Предобученный T5-трансформер временных рядов для Zero-Shot прогнозирования.

#### Рабочий код:
```python
# pip install aeon
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket

# Массив временных рядов: (N_samples, Channels, Length)
X_series = np.random.randn(50, 1, 300)

minirocket = MiniRocket(num_kernels=84, random_state=42)
X_features = minirocket.fit_transform(X_series)  # (50, 84) — готовы для подачи в LightGBM!
```

---

### 2.6 Табличные данные (Tabular Data)

Для редких категорий высокой кардинальности (High-Cardinality: почтовые индексы, ID клиентов, профессии):

1. **CatBoost / Target Encoding (`category_encoders`):**
   * Превращает категорию с 10,000 уникальных значений в 1–2 сглаженных вещественных числа.
2. **Entity Embeddings (PyTorch `nn.Embedding`):**
   * Обучение неглубокого слоя эмбеддингов размерности $d \approx \min(50, \text{num\_classes} // 2)$.
3. **`TabPFN`:**
   * Нейросеть, выполняющая байесовский инференс на таблицах до 1000 строк за секунду без обучения.

---

## 3. Продвинутые техники оптимизации

### 1. Matryoshka Representation Learning (MRL)
Современные модели (например, `nomic-embed`, `EmbeddingGemma`) обучаются так, чтобы **первые N чисел вектора содержали максимальную энергию сигнала**.
```python
full_emb = model.encode("Текст документа")  # (768,)
fast_emb = full_emb[:128]                    # Сжатие индекса в 6 раз с потерей качества < 1.2%!
```

### 2. Экспорт в ONNX Runtime + INT8 Квантование
Любую модель из PyTorch / Hugging Face можно конвертировать в формат ONNX:
* Потребление оперативной памяти падает в **3–4 раза**.
* Скорость инференса на CPU возрастает в **2–3 раза** благодаря векторным инструкциям AVX-512 / VNNI.

---

## 4. Сводная матрица выбора инженера

```
Какая у вас задача?
│
├── ТЕКСТ:
│   ├── Английский, высокая скорость CPU ────► all-MiniLM-L6-v2 (80 МБ)
│   ├── Русский язык / Мультиязычность ──────► multilingual-e5-small (140 МБ)
│   └── Длинные документы (>512 токенов) ────► nomic-embed-text-v1.5 (130 МБ)
│
├── ИЗОБРАЖЕНИЯ:
│   ├── Реалтайм, слабый процессор, IoT ─────► MobileNetV3 / V4 (9-15 МБ)
│   └── Максимум точности без обучения ──────► DINOv2-Small (85 МБ)
│
├── МУЛЬТИМОДАЛЬНЫЕ (ТЕКСТ + ФОТО):
│   └── Поиск по смыслу, Zero-Shot ──────────► MobileCLIP-S0 (35 МБ) / SigLIP (180 МБ)
│
├── ЗВУК И ГОЛОС:
│   ├── Речь, транскрипция, эмоции ──────────► Whisper-tiny Encoder (75 МБ)
│   └── Окружающие звуки и шумы ─────────────► CLAP-small (110 МБ)
│
└── ВРЕМЕННЫЕ РЯДЫ И ДАТЧИКИ:
    └── Скорость за доли секунды ────────────► MiniRocket (84 признака) / Catch22
```
