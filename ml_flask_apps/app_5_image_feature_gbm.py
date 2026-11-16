"""
===============================================================================
FLASK USE CASE 5: ГИБРИДНЫЙ ПАЙПЛАЙН (PRETRAINED CNN + LIGHTGBM)
===============================================================================

1. В ЧЕМ СУТЬ И ПОЧЕМУ ЭТО ОДИН ИЗ ГЛАВНЫХ ПАТТЕРНОВ СОВРЕМЕННОГО ML?
----------------------------------------------------------------------
Часто перед инженером встает задача классификации или предсказания по изображениям,
но:
  - Вычислительных ресурсов (GPU) мало или нет вовсе для дообучения (Fine-Tuning) тяжелых сетей.
  - Обучающая выборка маленькая (всего десятки или сотни примеров вместо сотен тысяч).
  - Задача мультимодальная (картинка + метаданные: цена, категория, дата).

Решение: ГИБРИДНАЯ АРХИТЕКТУРА (Frozen Backbone + GBDT Head):
  [Входное фото]
       │
       ▼ (Шаг 1: Предобработка: Resize 224x224, Normalization)
  [Предобученная нейросеть (Backbone: MobileNetV3 / ResNet / CLIP)]
       │
       ▼ (Шаг 2: Извлечение вектора признаков / Embeddings, 576 чисел)
  [L2 Нормализация + Метод главных компонент (PCA: сжатие до 12-32 компонент)]
       │
       ▼ (Шаг 3: Градиентный бустинг LightGBM / CatBoost)
  [Финальное предсказание класса и вероятностей]

2. КАРТА МОДЕЛЕЙ ДЛЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ (FEATURE EXTRACTORS):
--------------------------------------------------------------
  - Классические CNN: ResNet (ResNet-18/50), EfficientNet, MobileNetV3 (быстрый и легкий, ~9MB).
  - Vision Transformers (ViT): ViT-B/16, Swin Transformer (отличный глобальный контекст).
  - Self-Supervised (Короли эмбеддингов без учителя): DINOv2 от Meta.
  - Мультимодальные: CLIP от OpenAI (согласованное пространство текста и изображений).

3. ПОДВОДНЫЕ КАМНИ И БУТЫЛОЧНЫЕ ГОРЛЫШКИ (BOTTLENECKS):
------------------------------------------------------
  * Несоответствие геометрии (Geometry Mismatch):
    Эмбеддинги живут в непрерывном метрическом пространстве (косинусные расстояния, скалярные произведения).
    Деревья же делают прямоугольные ортогональные разбиения (x_i > порог).
    -> РЕШЕНИЕ: всегда применять L2-нормализацию и PCA перед подачей в бустинг!
  * Скорость инференса:
    LightGBM отвечает за 0.5 мс, но нейросеть на CPU тратит 15-50 мс на картинку.
  * Сдвиг домена (Domain Shift):
    ImageNet обучен на кошках, собаках, машинах. Для снимков МРТ или спутников веса ImageNet
    могут выделять не те паттерны, которые критичны для вашей задачи.

4. КАК ЗАПУСТИТЬ:
-----------------
    python app_5_image_feature_gbm.py
Затем откройте в браузере: http://127.0.0.1:5005/
===============================================================================
"""

import os
import io
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PyTorch и Torchvision для извлечения визуальных признаков
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights

# Scikit-Learn и LightGBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "vision_gbm_pipeline.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

from flask import Flask, request, render_template_string

app = Flask(__name__)

# Классы нашей задачи: классификация типов ландшафта / сцен
CLASSES = [
    "Лес и густая растительность (Forest)",
    "Водные просторы и побережье (Water)",
    "Песчаная пустыня и каньоны (Desert)",
    "Городская застройка и архитектура (Urban)"
]
CLASS_EMOJIS = ["🌲", "🌊", "🏜️", "🏙️"]


# =============================================================================
# ШАГ 1: ПОДГОТОВКА НЕЙРОСЕТИ ДЛЯ ИЗВЛЕЧЕНИЯ ЭМБЕДДИНГОВ (BACKBONE)
# =============================================================================
print("--> [Инициализация] Загрузка легковесной предобученной модели MobileNetV3...")

device = torch.device("cpu")  # Работаем на CPU для универсальности
weights = MobileNet_V3_Small_Weights.DEFAULT
feature_extractor = models.mobilenet_v3_small(weights=weights)

# Превращаем классификатор в Identity, чтобы модель отдавала сырой вектор признаков (576 чисел)
feature_extractor.classifier = torch.nn.Identity()
feature_extractor.eval()  # Переводим в режим инференса (отключаем Dropout и BatchNorm)

# Стандартный пайплайн подготовки картинки ImageNet: Resize 224x224, CenterCrop, Normalization
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features_from_pil(image: Image.Image) -> np.ndarray:
    """Прогоняет PIL-картинку через нейросеть и возвращает одномерный вектор (576,)."""
    rgb_img = image.convert('RGB')
    tensor = img_transform(rgb_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = feature_extractor(tensor)
    return embedding.cpu().numpy().flatten()


# =============================================================================
# ШАГ 2: ГЕНЕРАЦИЯ ОБРАЗЦОВ ДЛЯ ОБУЧЕНИЯ И СБОРКА ПАЙПЛАЙНА LIGHTGBM
# =============================================================================
def generate_synthetic_scene_image(class_idx: int) -> Image.Image:
    """Генерирует процедурные изображения с текстурами и палитрами каждого класса."""
    img = Image.new("RGB", (224, 224))
    draw = ImageDraw.Draw(img)

    if class_idx == 0:  # Лес (Зеленые оттенки, органические круги кроны)
        bg = (np.random.randint(20, 45), np.random.randint(70, 120), np.random.randint(20, 50))
        draw.rectangle([0, 0, 224, 224], fill=bg)
        for _ in range(35):
            x, y = np.random.randint(0, 224), np.random.randint(0, 224)
            r = np.random.randint(15, 45)
            c = (np.random.randint(10, 35), np.random.randint(100, 180), np.random.randint(15, 60))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c)

    elif class_idx == 1:  # Вода (Сине-бирюзовые волны и градиенты)
        bg = (np.random.randint(15, 40), np.random.randint(60, 110), np.random.randint(140, 210))
        draw.rectangle([0, 0, 224, 224], fill=bg)
        for i in range(12):
            y = i * 20 + np.random.randint(-5, 5)
            c = (np.random.randint(40, 80), np.random.randint(120, 190), np.random.randint(200, 255))
            draw.line([(0, y), (224, y + np.random.randint(-10, 10))], fill=c, width=np.random.randint(4, 10))

    elif class_idx == 2:  # Пустыня (Песочно-золотистые полосы дюн)
        bg = (np.random.randint(210, 245), np.random.randint(170, 205), np.random.randint(90, 130))
        draw.rectangle([0, 0, 224, 224], fill=bg)
        for _ in range(15):
            x0 = np.random.randint(-50, 224)
            y0 = np.random.randint(0, 224)
            c = (np.random.randint(180, 215), np.random.randint(130, 165), np.random.randint(60, 95))
            draw.arc([x0, y0, x0 + 180, y0 + 100], start=0, end=180, fill=c, width=6)

    elif class_idx == 3:  # Город (Серые фасады, сетка окон и дорог)
        bg = (np.random.randint(120, 160), np.random.randint(120, 160), np.random.randint(130, 170))
        draw.rectangle([0, 0, 224, 224], fill=bg)
        # Рисуем блоки зданий с прямоугольными окнами
        for bx in range(0, 224, 45):
            h = np.random.randint(80, 200)
            wall_color = (np.random.randint(60, 100), np.random.randint(60, 100), np.random.randint(70, 110))
            draw.rectangle([bx, 224 - h, bx + 38, 224], fill=wall_color)
            for wx in range(bx + 4, bx + 34, 10):
                for wy in range(224 - h + 10, 210, 14):
                    draw.rectangle([wx, wy, wx + 6, wy + 8], fill=(230, 230, 180))

    return img


def train_and_save_vision_gbm():
    print("--> [Обучение] Извлечение эмбеддингов для обучающей выборки сцен...")
    np.random.seed(42)
    torch.manual_seed(42)

    X_list = []
    y_list = []

    # Генерируем по 20 примеров на каждый класс (всего 80 изображений)
    samples_per_class = 20
    for cls_idx in range(4):
        for _ in range(samples_per_class):
            img = generate_synthetic_scene_image(cls_idx)
            emb = extract_features_from_pil(img)
            X_list.append(emb)
            y_list.append(cls_idx)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"--> Размер матрицы эмбеддингов: {X.shape} (80 картинок по 576 признаков)")

    # Сквозной пайплайн: L2-Нормализация -> PCA (сжатие до 12 главных осей) -> LightGBM
    pipeline = Pipeline([
        ('norm', Normalizer(norm='l2')),
        ('pca', PCA(n_components=12, random_state=42)),
        ('gbm', lgb.LGBMClassifier(
            n_estimators=60,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
            verbose=-1
        ))
    ])

    pipeline.fit(X, y)

    # Сохраняем модель вместе со сжатыми координатами центров для визуализации
    pca = pipeline.named_steps['pca']
    norm = pipeline.named_steps['norm']
    X_pca = pca.transform(norm.transform(X))

    payload = {
        'pipeline': pipeline,
        'X_pca_train': X_pca,
        'y_train': y
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"--> [Готово] Пайплайн обучен и сохранен в: {MODEL_PATH}")
    return payload


# Инициализация модели
if not MODEL_PATH.exists():
    saved_data = train_and_save_vision_gbm()
else:
    print(f"--> Загружаем готовую модель из {MODEL_PATH}")
    saved_data = joblib.load(MODEL_PATH)

vision_pipeline = saved_data['pipeline']
X_pca_train = saved_data['X_pca_train']
y_train = saved_data['y_train']


# Предварительно сгенерируем 4 образца для кнопок быстрого теста
SAMPLE_DATA_URIS = {}
for i in range(4):
    sample_img = generate_synthetic_scene_image(i)
    buffered = io.BytesIO()
    sample_img.save(buffered, format="JPEG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    SAMPLE_DATA_URIS[i] = f"data:image/jpeg;base64,{b64}"


# =============================================================================
# ШАГ 3: ФУНКЦИЯ ПОСТРОЕНИЯ 2D PCA ГРАФИКА (EDA ВЕКТОРОВ В РЕАЛЬНОМ ВРЕМЕНИ)
# =============================================================================
def generate_pca_scatter_plot(query_pca_2d: np.ndarray) -> str:
    """Строит 2D проекцию обучающих кластеров и новой тестовой картинки."""
    plt.figure(figsize=(7, 4.5), dpi=100)
    colors = ['#16a34a', '#0284c7', '#d97706', '#64748b']

    for cls_idx in range(4):
        mask = (y_train == cls_idx)
        plt.scatter(
            X_pca_train[mask, 0], X_pca_train[mask, 1],
            c=colors[cls_idx], label=CLASSES[cls_idx].split('(')[0].strip(),
            alpha=0.6, edgecolors='none', s=50
        )

    # Рисуем тестовую точку звездой
    plt.scatter(
        query_pca_2d[0], query_pca_2d[1],
        c='#dc2626', marker='*', s=250, label='Ваше изображение (Query)',
        edgecolors='black', linewidths=1.5, zorder=5
    )

    plt.title("2D Проекция пространства эмбеддингов (PCA Компоненты 1 и 2)", fontsize=11)
    plt.xlabel("Главная компонента 1 (PCA 1)", fontsize=10)
    plt.ylabel("Главная компонента 2 (PCA 2)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# =============================================================================
# ШАГ 4: HTML-ШАБЛОН СТРАНИЦЫ
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Pretrained Vision + LightGBM</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 25px; }
        .container { max-width: 950px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }
        h1 { color: #0f172a; margin-top: 0; font-size: 24px; }
        .desc { color: #64748b; font-size: 14px; margin-bottom: 25px; line-height: 1.5; }
        
        .upload-section { border: 2px dashed #cbd5e1; border-radius: 8px; padding: 25px; text-align: center; background: #f8fafc; }
        .upload-section:hover { border-color: #3b82f6; background: #f0fdf4; }
        .btn-classify { background: #2563eb; color: #fff; border: none; padding: 12px 24px; font-size: 15px; font-weight: bold; border-radius: 6px; cursor: pointer; margin-top: 10px; }
        .btn-classify:hover { background: #1d4ed8; }
        
        .quick-samples { margin-top: 15px; text-align: left; }
        .quick-grid { display: flex; gap: 12px; margin-top: 8px; }
        .sample-card { border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; cursor: pointer; text-align: center; background: #fff; flex: 1; transition: transform 0.15s, border-color 0.15s; }
        .sample-card:hover { transform: translateY(-2px); border-color: #2563eb; box-shadow: 0 2px 8px rgba(37,99,235,0.15); }
        .sample-card img { width: 100%; height: 75px; object-fit: cover; border-radius: 4px; margin-bottom: 4px; }
        
        .result-grid { display: grid; grid-template-columns: 280px 1fr; gap: 25px; margin-top: 30px; }
        .image-preview-card { background: #f1f5f9; padding: 15px; border-radius: 8px; text-align: center; }
        .image-preview-card img { max-width: 100%; height: auto; max-height: 220px; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
        
        .prediction-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; }
        .top-class { font-size: 22px; font-weight: bold; color: #1e293b; margin-bottom: 5px; }
        .confidence-badge { display: inline-block; padding: 4px 10px; border-radius: 14px; background: #dcfce7; color: #166534; font-weight: bold; font-size: 13px; margin-bottom: 15px; }
        
        .prob-row { margin-bottom: 8px; font-size: 13px; }
        .prob-bar-bg { background: #e2e8f0; border-radius: 4px; height: 10px; width: 100%; margin-top: 3px; overflow: hidden; }
        .prob-bar-fill { background: #3b82f6; height: 100%; }
        
        .vector-preview { background: #0f172a; color: #38bdf8; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 12px; margin-top: 15px; overflow-x: auto; white-space: nowrap; }
        
        .eda-section { margin-top: 25px; border-top: 1px solid #e2e8f0; padding-top: 20px; }
        .info-panel { margin-top: 30px; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 18px; font-size: 13px; color: #1e40af; line-height: 1.6; }
    </style>
</head>
<body>

<div class="container">
    <h1>🖼️ Гибридный инференс: MobileNetV3 (Feature Extractor) + LightGBM</h1>
    <div class="desc">
        Демонстрация сквозного паттерна: глубокая сверточная нейросеть извлекает <b>576 непрерывных признаков (эмбеддингов)</b>, 
        а компактный табличный бустинг (LightGBM) выполняет классификацию за <b>&lt; 1 миллисекунды</b>.
    </div>

    <div class="upload-section">
        <form method="POST" enctype="multipart/form-data">
            <input type="hidden" name="sample_idx" id="sample_idx" value="">
            
            <b>Загрузите свое изображение (JPG/PNG/WEBP):</b><br>
            <input type="file" name="image_file" accept="image/*"><br>
            <button type="submit" class="btn-classify">⚡ Извлечь признаки и классифицировать</button>
        </form>

        <div class="quick-samples">
            <span style="font-size: 13px; font-weight: 600; color: #475569;">Или выберите готовый образец для быстрого теста:</span>
            <div class="quick-grid">
                <div class="sample-card" onclick="submitSample(0)">
                    <img src="{{ sample_uris[0] }}" alt="Forest">
                    <span style="font-size: 12px; font-weight: 600;">🌲 Лес</span>
                </div>
                <div class="sample-card" onclick="submitSample(1)">
                    <img src="{{ sample_uris[1] }}" alt="Water">
                    <span style="font-size: 12px; font-weight: 600;">🌊 Вода</span>
                </div>
                <div class="sample-card" onclick="submitSample(2)">
                    <img src="{{ sample_uris[2] }}" alt="Desert">
                    <span style="font-size: 12px; font-weight: 600;">🏜️ Пустыня</span>
                </div>
                <div class="sample-card" onclick="submitSample(3)">
                    <img src="{{ sample_uris[3] }}" alt="Urban">
                    <span style="font-size: 12px; font-weight: 600;">🏙️ Город</span>
                </div>
            </div>
        </div>
    </div>

    {% if error %}
        <div style="background: #fee2e2; border: 1px solid #fca5a5; color: #991b1b; padding: 14px; border-radius: 6px; margin-top: 20px;">
            <b>Ошибка:</b> {{ error }}
        </div>
    {% endif %}

    {% if result %}
    <div class="result-grid">
        <div class="image-preview-card">
            <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; font-weight: 600;">ВХОДНОЕ ИЗОБРАЖЕНИЕ:</div>
            <img src="{{ result.img_data_uri }}" alt="Uploaded Image">
            <div style="margin-top: 10px; font-size: 12px; color: #475569;">
                Размер тензора: <code>[1, 3, 224, 224]</code><br>
                Эмбеддинг: <code>576 признаков</code>
            </div>
        </div>

        <div class="prediction-box">
            <div style="font-size: 12px; color: #64748b; font-weight: 600;">ПРЕДСКАЗАНИЕ LIGHTGBM:</div>
            <div class="top-class">{{ result.top_class }}</div>
            <span class="confidence-badge">Уверенность модели: {{ result.confidence }}%</span>

            <div style="margin-top: 10px;">
                <b>Вероятности по всем классам:</b>
                {% for item in result.all_probs %}
                <div class="prob-row">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{{ item.label }}</span>
                        <span><b>{{ item.prob }}%</b></span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {{ item.prob }}%;"></div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <div style="margin-top: 15px;">
                <b>Срез сырого вектора эмбеддинга (первые 10 измерений из 576):</b>
                <div class="vector-preview">{{ result.vector_preview }}</div>
            </div>
        </div>
    </div>

    <div class="eda-section">
        <h3>📊 EDA и пространство признаков в реальном времени</h3>
        <p style="font-size: 13px; color: #64748b; margin-top: -5px;">
            Эмбеддинги сжаты с помощью PCA (Principal Component Analysis). Красная звезда показывает, 
            в какой кластер обучающей выборки приземлилось ваше изображение:
        </p>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{{ result.pca_plot }}" style="max-width: 100%; border-radius: 8px; border: 1px solid #e2e8f0;">
        </div>
    </div>
    {% endif %}

    <div class="info-panel">
        <b>💡 Методология и устройство сквозного пайплайна:</b><br>
        1. <b>Нейросеть (Backbone):</b> Замораживает веса ImageNet и работает исключительно как «умный генератор признаков».<br>
        2. <b>L2-Нормализация:</b> Проецирует векторы на единичную сферу, чтобы угловые расстояния превратились в евклидовы.<br>
        3. <b>PCA (Сжатие размерности):</b> Снижает 576 измерений до главных осей дисперсии, предотвращая разреженность пространства для решающих деревьев.<br>
        4. <b>LightGBM Head:</b> Строит нелинейные границы разделения классов за доли миллисекунды без необходимости обучать сеть целиком.
    </div>
</div>

<script>
function submitSample(idx) {
    document.getElementById('sample_idx').value = idx;
    document.forms[0].submit();
}
</script>

</body>
</html>
"""


# =============================================================================
# ШАГ 5: ОБРАБОТЧИК ЗАПРОСОВ FLASK
# =============================================================================
@app.route('/', methods=['GET', 'POST'])
def vision_classifier():
    result = None
    error = None

    if request.method == 'POST':
        pil_image = None
        img_data_uri = None

        # Проверяем, нажал ли пользователь на один из готовых образцов
        sample_idx_str = request.form.get('sample_idx', '')
        if sample_idx_str != '' and sample_idx_str.isdigit():
            idx = int(sample_idx_str)
            if 0 <= idx < 4:
                pil_image = generate_synthetic_scene_image(idx)
                img_data_uri = SAMPLE_DATA_URIS[idx]

        # Иначе проверяем, был ли загружен файл через input file
        if pil_image is None and 'image_file' in request.files:
            file = request.files['image_file']
            if file and file.filename != '':
                try:
                    file_bytes = file.read()
                    pil_image = Image.open(io.BytesIO(file_bytes))
                    # Генерируем Data URI для отображения превью
                    b64 = base64.b64encode(file_bytes).decode('utf-8')
                    mime = "image/png" if file.filename.lower().endswith('.png') else "image/jpeg"
                    img_data_uri = f"data:{mime};base64,{b64}"
                except Exception as e:
                    error = f"Не удалось прочитать файл как изображение: {str(e)}"

        if pil_image is not None and error is None:
            try:
                # 1. Извлекаем 576-мерный эмбеддинг через MobileNetV3
                embedding = extract_features_from_pil(pil_image)

                # 2. Прогоняем через LightGBM пайплайн
                probs = vision_pipeline.predict_proba([embedding])[0]
                pred_class_idx = int(np.argmax(probs))
                confidence = round(float(probs[pred_class_idx]) * 100, 1)

                all_probs = [
                    {'label': f"{CLASS_EMOJIS[i]} {CLASSES[i]}", 'prob': round(float(probs[i]) * 100, 1)}
                    for i in range(4)
                ]
                all_probs = sorted(all_probs, key=lambda x: x['prob'], reverse=True)

                # 3. Рассчитываем координаты PCA для 2D-визуализации
                pca = vision_pipeline.named_steps['pca']
                norm = vision_pipeline.named_steps['norm']
                query_pca = pca.transform(norm.transform([embedding]))[0]

                pca_plot_b64 = generate_pca_scatter_plot(query_pca[:2])

                # Превью первых 10 компонент вектора
                vec_formatted = " ".join([f"{v:+.3f}" for v in embedding[:10]]) + " ... [576 values]"

                result = {
                    'top_class': f"{CLASS_EMOJIS[pred_class_idx]} {CLASSES[pred_class_idx]}",
                    'confidence': confidence,
                    'all_probs': all_probs,
                    'img_data_uri': img_data_uri,
                    'vector_preview': vec_formatted,
                    'pca_plot': pca_plot_b64
                }
            except Exception as e:
                error = f"Ошибка в ходе инференса: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        error=error,
        sample_uris=SAMPLE_DATA_URIS
    )


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Vision + LightGBM Hybrid Classifier запущен!")
    print("👉 Откройте в браузере: http://127.0.0.1:5005/")
    print("="*70 + "\n")
    app.run(debug=True, port=5005)
