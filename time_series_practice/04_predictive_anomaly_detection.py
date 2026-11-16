"""
================================================================================
ПРИМЕР 4: ПРОДВИНУТАЯ ДЕТЕКЦИЯ АНОМАЛИЙ С ИСПОЛЬЗОВАНИЕМ DEEP LEARNING (LSTM)
================================================================================

В этом примере мы реализуем один из самых мощных индустриальных методов детекции:
"Forecast-based Anomaly Detection" на базе рекуррентных нейронных сетей (LSTM).

ПОЧЕМУ ЭТО СЛОЖНЕЕ И ЛУЧШЕ:
1. Табличные методы (MAD, Isolation Forest) работают с "фичами", которые мы придумали сами (лаги, разности).
2. LSTM обучается понимать ВРЕМЕННУЮ ДИНАМИКУ СИСТЕМЫ напрямую:
   - Она сама выучивает, что после пика в 15:00 нагрузка в 16:00 будет спадать.
   - Если в 16:00 нагрузка ВДРУГ ВЫРОСЛА — LSTM предскажет спад (по выученной норме), а реальность покажет рост.
   - Ошибка прогноза (Residual) $|y_t - \hat{y}_t|$ будет колоссальной.
3. Динамический порог (Dynamic Thresholding):
   Мы не используем статический порог. Мы используем конформный порог (определяем порог по 98-му перцентилю ошибок на чистых данных), что делает модель адаптивной.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. СЕТЕВАЯ АРХИТЕКТУРА: LSTM ПРЕДИКТОР
# ==============================================================================
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hn, _) = self.lstm(x)
        # Берем только последний выход рекуррентной сети
        prediction = self.fc(hn[-1])
        return prediction


# ==============================================================================
# 2. ПОДГОТОВКА ДАННЫХ (SLIDING WINDOWS)
# ==============================================================================
def create_sequences(data, seq_len=24):
    """Превращает ряд в набор окон (batch, seq_len, 1)"""
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs).reshape(-1, seq_len, 1), np.array(ys)


# ==============================================================================
# 3. ОСНОВНОЙ ПАЙПЛАЙН
# ==============================================================================
def run_deep_anomaly_pipeline():
    # Используем данные из примера 03
    from importlib import import_module
    mod = import_module('03_anomaly_detection')
    df = mod.generate_server_metrics_with_anomalies()
    
    print("=" * 70)
    print("🚀 4. DEEP LEARNING (LSTM) ДЕТЕКЦИЯ АНОМАЛИЙ")
    print("=" * 70)

    # Нормализация (критично для нейросетей)
    scaler = StandardScaler()
    scaled_rps = scaler.fit_transform(df[['rps']]).flatten()

    # Создание последовательностей (окно 24 часа)
    seq_len = 24
    X, y = create_sequences(scaled_rps, seq_len)
    
    # Разделение на тренировку (первые 500 часов - чисто нормальные)
    train_size = 500
    X_train, y_train = torch.FloatTensor(X[:train_size]), torch.FloatTensor(y[:train_size])
    X_all, y_all = torch.FloatTensor(X), torch.FloatTensor(y)

    # Обучение модели
    model = LSTMForecaster()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Обучение LSTM предиктора (30 эпох)...")
    for epoch in range(30):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f"Обучение завершено. Финальный Loss: {loss.item():.4f}")

    # Детекция аномалий: вычисление ошибки прогноза на всем ряду
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_all).numpy().flatten()
    
    # Ошибка (в исходном масштабе)
    y_actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    preds_actual = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    residuals = np.abs(y_actual - preds_actual)

    # Динамический квантильный порог (98% ошибок считаем нормой)
    threshold = np.quantile(residuals[:train_size], 0.98)
    is_anomaly = (residuals > threshold).astype(int)

    # Подгонка размерности для DataFrame
    full_is_anomaly = pd.Series(0, index=df.index)
    full_is_anomaly.iloc[seq_len:] = is_anomaly
    
    # Визуализация
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index[seq_len:], y_actual, label="Факт (RPS)", color="#34495e")
    plt.plot(df.index[seq_len:], preds_actual, label="Прогноз LSTM (Ожидаемая норма)", color="#27ae60", alpha=0.8)
    plt.title("LSTM Predictor: Ожидаемая норма vs Реальность", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(df.index[seq_len:], residuals, label="Ошибка прогноза (Residual)", color="#e67e22")
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Порог аномалии (98% quantile = {threshold:.1f})")
    true_anoms = df[df["is_anomaly"] == 1]
    plt.scatter(true_anoms.index, [0]*len(true_anoms), color="red", marker="|", label="Истинные аномалии")
    
    plt.title("Ошибка прогноза и адаптивный порог", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("plots/04_deep_predictive_anomaly.png", dpi=200)
    print("📊 График сохранен в plots/04_deep_predictive_anomaly.png")

if __name__ == "__main__":
    run_deep_anomaly_pipeline()
