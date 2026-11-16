# 🚀 INTER_IT: Applied Machine Learning, Pipelines & Full-Stack Showcase

Комплексный репозиторий прикладного машинного обучения: от физически-информированных моделей переноса загрязнителей и анализа временных рядов до пошагового обучающего курса и полнофункциональных веб-приложений на Flask.

---

## 📂 Структура репозитория

```text
INTER_IT/
├── 📘 FEATURE_EXTRACTION_GUIDE.md     # 🌟 Справочник по компактным экстракторам признаков
│
├── ⏱️ time_series_practice/           # 🕒 Практикум: Анализ и прогнозирование временных рядов
│   ├── README.md                      # Теория, архитектура фичей и метрики
│   ├── 01_short_term_forecasting.py   # Краткосрочный прогноз (LightGBM, One-Step vs Recursive)
│   ├── 02_long_term_forecasting.py    # Долгосрочный прогноз на 90 дней (Фурье + STL + Коридор 90%)
│   ├── 03_anomaly_detection.py        # Детекция аномалий (Seasonal MAD, STL, Isolation Forest)
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

## ⏱️ 1. Практикум по временным рядам (`time_series_practice/`)

Подробное руководство и исходный код: [`time_series_practice/README.md`](./time_series_practice/README.md).

* **`01_short_term_forecasting.py`**: Прогноз следующих шагов (1–48 часов). Формирование лагов (`lag_1`, `lag_24`), скользящих статистик (без утечки данных через `.shift(1)`), циклических признаков синуса/косинуса. Сравнение One-Step прогноза и рекурсивного прогноза.
* **`02_long_term_forecasting.py`**: Прогноз на длинный горизонт (90 дней вперед). Почему рекурсивные короткие лаги накапливают ошибку и как строить прогноз на базе декомпозиции STL, гармоник Фурье и детерминированных календарных признаков с 90%-м коридором неопределенности.
* **`03_anomaly_detection.py`**: Детекция инцидентов и выбросов (точечные всплески, скрытые ночные контекстные аномалии, сдвиг уровня). Реализация 3 методов: Seasonal Rolling MAD, анализ остатков STL и Isolation Forest на мультифичах.

---

## 🌊 2. Задача переноса загрязнителя (`chemical_dispersion_task/`)

Моделируется прорыв заглубленного резервуара с трихлорэтиленом (TCE) в песчаном водоносном горизонте. Загрязнитель переносится грунтовыми водами по закону Дарси в сторону городского водозабора.

Подробная документация: [`chemical_dispersion_task/README.md`](./chemical_dispersion_task/README.md).

### 4 требуемых выхода единого пайплайна:
1. **Точечный прогноз (`c_pred`):** Концентрация TCE ($\text{мкг/л}$) для каждой точки $(x, y, z, t)$ (минимизация RMSE).
2. **Границы уверенности (`[c_lower, c_upper]`):** Калиброванный 90%-й доверительный интервал (Conformalized Quantile Regression).
3. **Масса яда на 01.01.2026 (`mass_2026`):** Суммарная масса растворенного TCE во всем водоносном бассейне в кг.
4. **Дата пробоя водозабора (`arrival_time`):** Момент времени, когда концентрация на водозаборе впервые пробьет порог **5.0 мкг/л** (ПДК).

---

## 📚 3. Обучающий экспресс-курс по ML (`ml_tutorials/`)

Каждый файл в папке [`ml_tutorials/`](./ml_tutorials/) полностью автономен, содержит подробные комментарии для начинающих, генерирует реалистичный датасет, обучает модель, считает метрики и сохраняет графики в `plots/`:

* **`01_linear_regression.py`**: Основы регрессии, веса $w$ и смещение $b$, метрики MAE, RMSE, $R^2$.
* **`02_logistic_regression.py`**: Бинарная классификация, сигмоида, нормализация, Confusion Matrix, ROC-AUC.
* **`03_decision_tree.py`**: Нелинейные правила, критерий Джини, наглядная демонстрация переобучения (`max_depth`).
* **`04_random_forest.py`**: Ансамбли (Бэггинг), параллелизм на CPU (`n_jobs=-1`), оценка важности признаков.
* **`05_gradient_boosting.py`**: Последовательное исправление ошибок (LightGBM), Early Stopping, кривые обучения.
* **`06_kmeans_clustering.py`**: Обучение без учителя, геометрические центроиды, подбор $K$ (Elbow Method & Silhouette).
* **`07_production_pipeline.py`**: Промышленная сборка `ColumnTransformer` (числа $\to$ Imputer + Scaler; категории $\to$ OneHotEncoder).

---

## 🌐 4. Витрина веб-приложений на Flask (`ml_flask_apps/`)

Коллекция нетривиальных прикладных сценариев, демонстрирующих инференс ML в реальном веб-интерфейсе:

1. **AI Support Ticket Router:** NLP-анализ неструктурированного текста (`TfidfVectorizer` + `LogisticRegression`), маршрутизация тикетов по отделам, расчет вероятности и приоритета SLA.
2. **Explainable Health Risk Screener:** Объяснимый ИИ (XAI) для кардиологического риска. Разложение вклада каждого клинического фактора в балл риска + симулятор «What-If».
3. **Dynamic Real Estate & ROI Simulator:** Табличный бустинг для оценки жилья с мгновенным расчетом окупаемости инвестиций в евроремонт.
4. **Enterprise Fraud & Anomaly Audit:** Загрузка файлов CSV, потоковая обработка в памяти без записи на диск, обучение без учителя (`IsolationForest`).
5. **Computer Vision + LightGBM Hybrid:** Предобученная сверточная сеть (`MobileNetV3`) извлекает 576 непрерывных признаков, а L2-нормализатор, PCA и LightGBM мгновенно классифицируют ландшафт с 2D проекцией EDA.
6. **Intelligent Document OCR Pipeline:** 4-фазный пайплайн распознавания документов: адаптивная бинаризация OpenCV, разметка рамок слов (Bounding Boxes), чтение текста на русском и английском (Tesseract LSTM) и парсинг сущностей регулярными выражениями.
7. **Local Transformer NLP Pipeline:** Запуск компактной модели `sentence-transformers/all-MiniLM-L6-v2` (~80 МБ) на CPU: токенизация WordPiece, Mean Pooling с Attention Mask, извлечение 384-мерного эмбеддинга, классификация и демонстрация Matryoshka Representation Learning (MRL).

---

## 📘 5. Справочник по экстракторам признаков (`FEATURE_EXTRACTION_GUIDE.md`)

В репозитории доступен подробный справочник [**`FEATURE_EXTRACTION_GUIDE.md`**](./FEATURE_EXTRACTION_GUIDE.md) по выбору компактных локальных моделей (весом 10–150 МБ) для извлечения признаков из текста, изображений, мультимодальных пар, аудио, временных рядов и таблиц.

---

## 🛠️ Установка и быстрый запуск

### 1. Клонирование и установка зависимостей:
```bash
git clone https://github.com/Steklish/INTER_IT.git
cd INTER_IT
pip install -r requirements.txt
```

### 2. Запуск модуля временных рядов:
```bash
cd time_series_practice
python 01_short_term_forecasting.py  # Краткосрочный прогноз
python 02_long_term_forecasting.py   # Долгосрочный прогноз (90 дней)
python 03_anomaly_detection.py       # Детекция аномалий
```

### 3. Запуск единого портала веб-приложений:
```bash
cd ml_flask_apps
python portal.py
```
👉 Откройте в браузере: **`http://127.0.0.1:5000/`**.
