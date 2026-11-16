# ⏱️ Практикум по машинному обучению на временных рядах (Time Series Machine Learning)

Комплексный учебно-практический модуль, посвященный четырем ключевым задачам анализа временных рядов:
1. **Краткосрочный прогноз следующих шагов (Short-Term / Next Steps)** с лагами и скользящими окнами.
2. **Долгосрочный прогноз на большой горизонт (Long-Term / Multi-Horizon)** с декомпозицией и гармониками Фурье.
3. **Детекция аномалий (Time Series Anomaly Detection)**: классические методы (MAD, STL, Isolation Forest).
4. **Deep Learning: Продвинутая детекция аномалий (Predictive Forecasting Model)**: LSTM-предиктор с динамическим порогом.

---

## 📂 Структура модуля

```text
time_series_practice/
├── 📄 README.md                      # Теоретическое руководство и архитектурные рекомендации
├── ⚡ 01_short_term_forecasting.py    # 1. Почасовой прогноз энергопотребления (LightGBM, One-Step vs Recursive)
├── 📈 02_long_term_forecasting.py     # 2. Долгосрочный прогноз спроса на 90 дней (STL + Фурье + Коридор 90%)
├── 🔍 03_anomaly_detection.py         # 3. Детекция аномалий (Seasonal MAD, STL, Isolation Forest, LightGBM Pred)
├── 🧠 04_predictive_anomaly_detection.py # 4. Advanced Deep Learning (PyTorch LSTM Forecaster + Dynamic Thresholding)
└── 📊 plots/                          # Сгенерированные аналитические графики
    ├── 01_short_term_forecast.png
    ├── 02_long_term_forecast.png
    ├── 03_anomaly_detection.png
    └── 04_deep_predictive_anomaly.png
```

---

## ⚡ 1. Краткосрочный прогноз следующих шагов (`01_short_term_forecasting.py`)

(Содержание сокращено - см. код модуля)

---

## 📈 2. Долгосрочный прогноз на большой горизонт (`02_long_term_forecasting.py`)

(Содержание сокращено - см. код модуля)

---

## 🔍 3. Детекция аномалий: Классические методы (`03_anomaly_detection.py`)

Автоматическое обнаружение инцидентов (точечные всплески, контекстные ночные аномалии, сдвиги режимов) с помощью:
1. **Seasonal-Adjusted Rolling MAD** (робастный коридор).
2. **STL Residual Analysis** (анализ остатков после вычитания цикла).
3. **Isolation Forest** (ML без учителя).
4. **LightGBM Predictor** (табличный предиктор ошибок).

---

## 🧠 4. Advanced: Deep Learning (LSTM) Детекция (`04_predictive_anomaly_detection.py`)

### Задача
Обнаружение аномалий с использованием обучаемой **предсказывающей модели (Predictive Forecasting Model)** на базе LSTM (Long Short-Term Memory).

### Механика работы
1. **Обучение предиктора**: Нейросеть учится предсказывать следующее значение $\hat{y}_t$ на основе истории $[y_{t-W}, \dots, y_{t-1}]$.
2. **Анализ ошибки (Residuals)**: Мы вычисляем "удивление" модели: $e_t = |y_t - \hat{y}_t|$.
3. **Адаптивный порог**: Аномалия помечается, если ошибка превышает динамически вычисленный квантиль (напр. 98%) на валидационных данных.

Этот подход позволяет эффективно ловить сложные контекстные аномалии, которые невозможно описать простыми статистическими правилами.

### Результат:
График [`time_series_practice/plots/04_deep_predictive_anomaly.png`](./plots/04_deep_predictive_anomaly.png) наглядно показывает, как ошибка прогноза взлетает при аномалиях, превышая адаптивный порог.

---

## 🚀 Команды для запуска

Перейдите в каталог `time_series_practice`:

```bash
cd time_series_practice
```

### 1-3. Базовые методы:
```bash
python 01_short_term_forecasting.py
python 02_long_term_forecasting.py
python 03_anomaly_detection.py
```

### 4. Advanced Deep Learning Anomaly Detection:
```bash
python 04_predictive_anomaly_detection.py
```
