# 🚀 INTER_IT: Applied Machine Learning, Pipelines & Full-Stack Showcase

Комплексный репозиторий прикладного машинного обучения: от физически-информированных моделей переноса загрязнителей и анализа временных рядов до автоэнкодеров, образовательного курса и полнофункциональных веб-приложений на Flask.

---

## 📂 Структура репозитория

```text
INTER_IT/
├── 📘 FEATURE_EXTRACTION_GUIDE.md     # 🌟 Справочник по компактным экстракторам признаков
│
├── 🎭 autoencoder_practice/           # 🧬 Практикум по автоэнкодерам (4 прикладных сценария)
│   ├── README.md                      # Теория, архитектуры (DAE, VAE) и метрики
│   ├── 01_dimensionality_reduction.py # Нелинейное сжатие данных (64D -> 2D, превосходство над PCA)
│   ├── 02_denoising_autoencoder.py    # Шумоподавление сигналов (Conv1D DAE, прирост +12 dB)
│   ├── 03_anomaly_detection_ae.py     # Детекция аварий оборудования (ROC-AUC = 1.0, Root Cause)
│   ├── 04_variational_autoencoder.py  # VAE: генерация новых объектов и плавный морфинг
│   └── plots/                         # Сгенерированные аналитические графики высокого разрешения
│
├── ⏱️ time_series_practice/           # 🕒 Практикум: Анализ и прогнозирование временных рядов
│   ├── README.md                      # Теория, архитектура фичей и метрики
│   ├── 01_short_term_forecasting.py   # Краткосрочный прогноз (LightGBM, One-Step vs Recursive)
│   ├── 02_long_term_forecasting.py    # Долгосрочный прогноз на 90 дней (Фурье + STL + Коридор 90%)
│   ├── 03_anomaly_detection.py        # Детекция аномалий (Seasonal MAD, STL, Isolation Forest, LightGBM)
│   ├── 04_predictive_anomaly_detection.py # Advanced Deep Learning (PyTorch LSTM + Dynamic Threshold)
│   └── plots/                         # Сгенерированные графики прогнозов и аномалий
│
├── 🌊 chemical_dispersion_task/       # 🧪 Задача прогнозирования распространения химиката (TCE)
│   ├── README.md                      # Документация и руководство по запуску решений
│   ├── TASK_SPECIFICATION.md          # Формальная спецификация и физика процесса
│   ├── solution.py                    # 🏆 Боевой конвейер LightGBM + Conformal Prediction
│   ├── baseline.py                    # Базовое решение IDW
│   ├── solution_gpr.py                # Гауссовские процессы (GPR)
│   ├── solution_pinn.py               # Физически-информированная нейросеть (PINN)
│   ├── solution_numsim.py             # Численная гидрогеологическая симуляция
│   ├── scorer.py                      # Валидация и расчет метрик жюри
│   ├── benchmark.py                   # Сравнительный бенчмарк всех 5 подходов
│   ├── visualize.py                   # 3D визуализация шлейфа загрязнения
│   ├── dashboard.html                 # Интерактивный веб-дашборд с 3D срезом
│   ├── data/                          # Наблюдательная сеть скважин и замеры
│   └── plots/                         # Графики метрик и динамики шлейфа
│
├── 📚 ml_tutorials/                   # 🎓 Пошаговый образовательный курс по алгоритмам ML
│   ├── ROADMAP.md                     # Дорожная карта и руководство по экспериментам
│   ├── 01_linear_regression.py        # Линейная регрессия (оценка недвижимости)
│   ├── 02_logistic_regression.py      # Логистическая регрессия (банковский скоринг)
│   ├── 03_decision_tree.py            # Дерево решений и переобучение (Overfitting)
│   ├── 04_random_forest.py            # Случайный лес и важность признаков
│   ├── 05_gradient_boosting.py        # Градиентный бустинг (LightGBM)
│   ├── 06_kmeans_clustering.py        # Кластеризация K-Means (сегментация клиентов)
│   ├── 07_production_pipeline.py      # Сквозной пайплайн ColumnTransformer
│   └── plots/                         # Сгенерированные визуализации и графики метрик
│
└── 🌐 ml_flask_apps/                  # 🚀 Интерактивные веб-сервисы машинного обучения на Flask
    ├── portal.py                      # ⭐ Единый мастер-портал всех 7 приложений (порт 5000)
    ├── app_1_nlp_ticket_router.py     # 📨 1. NLP Классификация обращений (TF-IDF + Softmax)
    ├── app_2_health_risk_screener.py  # ❤️ 2. Объяснимый риск (Feature Attribution & What-If)
    ├── app_3_dynamic_pricing.py       # 🏢 3. Оценка недвижимости и ROI ремонта
    ├── app_4_fraud_detector.py        # 🛡️ 4. Пакетный CSV-аудит и поиск аномалий (Isolation Forest)
    ├── app_5_image_feature_gbm.py     # 🖼️ 5. Компьютерное зрение (MobileNetV3 + LightGBM)
    ├── app_6_ocr_pipeline.py          # 📄 6. Интеллектуальный OCR-пайплайн документов (OpenCV + Tesseract)
    ├── app_7_transformer_pipeline.py  # 🤖 7. Локальный Трансформер NLP (MiniLM + Matryoshka MRL)
    ├── ocr_pipeline.py                # ⚙️ Модуль OCR: предобработка, Bounding Boxes, RegEx
    ├── transformer_pipeline.py        # ⚙️ Модуль Трансформера: Mean Pooling, MRL, классификация
    ├── data/                          # Тестовые датасеты для пакетного скоринга
    ├── models/                        # Сериализованные обученные модели (.joblib)
    └── tessdata/                      # Локальные языковые пакеты OCR
```

---

## 🎭 1. Практикум по автоэнкодерам (`autoencoder_practice/`)

Подробное руководство и исходный код: [`autoencoder_practice/README.md`](./autoencoder_practice/README.md).

* **`01_dimensionality_reduction.py`**: Сжатие 64-мерных данных сенсоров в 2D Bottleneck. Демонстрация превосходства нелинейного автоэнкодера над линейным PCA (MSE снижен на **68.0%**).
* **`02_denoising_autoencoder.py`**: Шумоподавляющий автоэнкодер (Conv1D DAE). Очистка биосигналов от электромагнитных и импульсных помех без срезания острых пиков (прирост чистоты сигнала **+12.0 dB**, MSE снижен на **93.1%**).
* **`03_anomaly_detection_ae.py`**: Полуавтоматическая детекция аварий турбины (обучение ТОЛЬКО на норме). Выявление аномалий по ошибке реконструкции ($\text{ROC-AUC} = 1.0$) и автоматическая локализация дефектного датчика (Root Cause Analysis).
* **`04_variational_autoencoder.py`**: Вариационный автоэнкодер (VAE). Трюк перепараметризации, оптимизация ELBO, генерация новых волновых форм из шума $\mathcal{N}(0, \mathbf{I})$ и непрерывный морфинг между режимами (Latent Walk).

---

## ⏱️ 2. Практикум по временным рядам (`time_series_practice/`)

Подробное руководство и исходный код: [`time_series_practice/README.md`](./time_series_practice/README.md).

* **`01_short_term_forecasting.py`**: Прогноз почасового энергопотребления на 1–48 часов (LightGBM, One-Step vs Recursive).
* **`02_long_term_forecasting.py`**: Прогноз на 90 дней вперед (STL + гармоники Фурье + 90% доверительный коридор).
* **`03_anomaly_detection.py`**: Детекция аномалий во временных рядах (Seasonal MAD, STL Residuals, Isolation Forest, LightGBM Predictor).
* **`04_predictive_anomaly_detection.py`**: Продвинутая детекция аномалий с помощью PyTorch LSTM предиктора и динамического порога.

---

## 🌊 3. Задача переноса загрязнителя (`chemical_dispersion_task/`)

Моделируется прорыв заглубленного резервуара с трихлорэтиленом (TCE) в песчаном водоносном горизонте. Загрязнитель переносится грунтовыми водами по закону Дарси в сторону городского водозабора.
Подробная документация: [`chemical_dispersion_task/README.md`](./chemical_dispersion_task/README.md).

---

## 📚 4. Обучающий экспресс-курс по ML (`ml_tutorials/`)

Каждый файл в папке [`ml_tutorials/`](./ml_tutorials/) полностью автономен, содержит подробные комментарии для начинающих: от линейной регрессии до сквозного пайплайна `ColumnTransformer`.

---

## 🌐 5. Витрина веб-приложений на Flask (`ml_flask_apps/`)

Коллекция 7 интерактивных сценариев: классификация тикетов, медицинский скоринг риска, оценка недвижимости, аудит мошенничества, Computer Vision + LightGBM, интеллектуальный OCR документов и локальный трансформер MiniLM.

---

## 🛠️ Установка и быстрый запуск

### 1. Клонирование и установка зависимостей:
```bash
git clone https://github.com/Steklish/INTER_IT.git
cd INTER_IT
pip install -r requirements.txt
```

### 2. Запуск модуля автоэнкодеров:
```bash
cd autoencoder_practice
python 01_dimensionality_reduction.py  # Сжатие данных (Autoencoder vs PCA)
python 02_denoising_autoencoder.py      # Очистка сигналов от шума (DAE)
python 03_anomaly_detection_ae.py       # Детекция аварий и Root Cause Analysis
python 04_variational_autoencoder.py    # VAE: генерация и латентный морфинг
```

### 3. Запуск модуля временных рядов:
```bash
cd time_series_practice
python 01_short_term_forecasting.py
python 02_long_term_forecasting.py
python 03_anomaly_detection.py
python 04_predictive_anomaly_detection.py
```

### 4. Запуск единого портала веб-приложений:
```bash
cd ml_flask_apps
python portal.py
```
👉 Откройте в браузере: **`http://127.0.0.1:5000/`**.
