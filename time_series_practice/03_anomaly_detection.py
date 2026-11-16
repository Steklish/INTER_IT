"""
================================================================================
ПРИМЕР 3: ДЕТЕКЦИЯ АНОМАЛИЙ ВО ВРЕМЕННЫХ РЯДАХ (TIME SERIES ANOMALY DETECTION)
================================================================================

В этой задаче мы находим нетипичные отклонения и инциденты во временном ряде.
Это критически важный инструмент для:
  - Мониторинга серверов и сетевой инфраструктуры (DevOps / SRE)
  - Детекции мошенничества и взломов (кибербезопасность / антифрод)
  - Предиктивного обслуживания промышленного оборудования и датчиков IoT

ТИПЫ АНОМАЛИЙ ВО ВРЕМЕННЫХ РЯДАХ:
1. Точечные выбросы (Point Spikes):
   Кратковременный импульсный скачок (DDoS-атака или сбой счетчика).
2. Контекстные / Сезонные аномалии (Contextual Anomalies):
   Значение абсолютно нормальное днем (например, 55 RPS), но аномальное ночью (в 04:00 норма 20 RPS).
   Глобальный статический порог (например, >80 RPS) такую аномалию гарантированно пропустит!
3. Сдвиг уровня / Смена режима (Level Shift):
   Внезапное застревание ряда на повышенном или пониженном уровне (утечка памяти или падение ноды).

ГЛАВНАЯ ЛОВУШКА ДЛЯ НАЧИНАЮЩИХ:
Если просто посчитать стандартное отклонение в скользящем окне за 24 часа по «сырому» ряду,
естественный перепад между днем (100) и ночью (20) раздует разброс (sigma ~ 30).
В результате порог поднимется до 160+, и ночные аномалии будут пропущены!

ПРАВИЛЬНЫЙ ИНДУСТРИАЛЬНЫЙ ПОДХОД:
Сначала вычитается ожидаемый суточный профиль (сезонная поправка), и статистический
коридор строится на основе отклонения от нормы конкретного часа суток.

3 МЕТОДА В ЭТОМ СКРИПТЕ:
1. Сезонно-скорректированный робастный коридор (Seasonal-Adjusted Rolling MAD).
2. Анализ остатков декомпозиции (STL Residual Analysis).
3. Обучение без учителя на многомерных признаках (Isolation Forest).
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from statsmodels.tsa.seasonal import seasonal_decompose


# ==============================================================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ МОНИТОРИНГА С 3 ВИДАМИ АНОМАЛИЙ
# ==============================================================================
def generate_server_metrics_with_anomalies(n_days=30, random_seed=42):
    """
    Генерирует почасовые метрики сетевого трафика / запросов к серверу (RPS) за 30 дней (720 точек).
    Включает суточный цикл, случайный шум и искусственно внедренные аномалии 3 типов:
      1. Point Spikes (импульсные всплески)
      2. Contextual Anomaly (ночной пик, имитирующий утечку данных / сканирование портов)
      3. Level Shift (сдвиг уровня на сутки из-за бага)
    """
    np.random.seed(random_seed)
    n_hours = n_days * 24
    timestamps = pd.date_range(start="2026-03-01 00:00:00", periods=n_hours, freq="h")

    # Нормальный суточный сигнал: пик днем (15:00) ~100 RPS, минимум ночью (04:00) ~20 RPS
    hours = timestamps.hour.values
    normal_signal = 60.0 + 40.0 * np.sin(2 * np.pi * (hours - 9) / 24)
    noise = np.random.normal(loc=0.0, scale=3.5, size=n_hours)
    
    traffic = normal_signal + noise
    labels = np.zeros(n_hours, dtype=int)  # 0 = норма, 1 = аномалия
    anomaly_types = ["Normal"] * n_hours

    # 1. Точечные всплески (Point Spikes) в часы 120, 310, 620
    spike_indices = [120, 310, 620]
    for idx in spike_indices:
        traffic[idx] += 95.0  # Мощный мгновенный выброс
        labels[idx] = 1
        anomaly_types[idx] = "Point Spike"

    # 2. Контекстная аномалия: ночной всплеск в 03:00–05:00 утра на 15-й день (часы 363-365)
    # Значение поднимается до 62 RPS. Днем 62 RPS — рядовая норма, но в 04:00 утра норма — 20 RPS!
    contextual_indices = [363, 364, 365]
    for idx in contextual_indices:
        traffic[idx] = 65.0 + np.random.normal(0, 1.5)
        labels[idx] = 1
        anomaly_types[idx] = "Contextual (Night Spike)"

    # 3. Сдвиг базового уровня (Level Shift) на 22-23 дни (часы 520-544)
    shift_indices = list(range(520, 545))
    for idx in shift_indices:
        traffic[idx] += 30.0  # Базовая нагрузка выросла из-за утечки памяти
        labels[idx] = 1
        anomaly_types[idx] = "Level Shift"

    df = pd.DataFrame({
        "timestamp": timestamps,
        "rps": traffic,
        "is_anomaly": labels,
        "anomaly_type": anomaly_types
    }).set_index("timestamp")

    return df


# ==============================================================================
# 2. МЕТОД 1: СЕЗОННО-СКОРРЕКТИРОВАННЫЙ РОБАСТНЫЙ КОРИДОР (SEASONAL ROLLING MAD)
# ==============================================================================
def detect_anomalies_seasonal_mad(series, threshold_sigma=3.0):
    """
    1. Вычисляет эталонный суточный профиль (медиана по каждому часу суток 0..23).
    2. Находит отклонение фактического значения от суточного профиля.
    3. Применяет скользящий MAD (Median Absolute Deviation) к этому отклонению.
    4. Строит динамический коридор [Lower_Bound, Upper_Bound], огибающий суточный график.
    """
    # 1. Эталонное значение для каждого часа дня
    hourly_baseline = series.groupby(series.index.hour).transform("median")
    deviation = series - hourly_baseline

    # 2. Скользящий робастный разброс отклонений
    window = 12
    rolling_mad = deviation.abs().rolling(window=window, center=True, min_periods=3).median()
    sigma_est = 1.4826 * rolling_mad
    sigma_est = sigma_est.bfill().ffill()  # Заполняем краевые эффекты

    lower_bound = hourly_baseline - threshold_sigma * sigma_est
    upper_bound = hourly_baseline + threshold_sigma * sigma_est

    is_anomaly = (series < lower_bound) | (series > upper_bound)
    return is_anomaly.astype(int), lower_bound, upper_bound


# ==============================================================================
# 3. МЕТОД 2: АНАЛИЗ ОСТАТКОВ ДЕКОМПОЗИЦИИ (STL RESIDUAL ANALYSIS)
# ==============================================================================
def detect_anomalies_stl(series, period=24, threshold_sigma=2.5):
    """
    Раскладывает ряд на Тренд, Сезонность и Остаток.
    Идеален для точечных всплесков и контекстных аномалий:
    при вычитании периодической составляющей ночной пик становится огромным всплеском в остатках.
    """
    decomp = seasonal_decompose(series, model="additive", period=period)
    residuals = decomp.resid.fillna(0.0)

    # Робастное стандартное отклонение остатков через MAD
    mad_res = (residuals - residuals.median()).abs().median()
    sigma_res = 1.4826 * mad_res

    is_anomaly = (residuals.abs() > threshold_sigma * sigma_res).astype(int)
    return is_anomaly, residuals, threshold_sigma * sigma_res


# ==============================================================================
# 4. МЕТОД 3: ОБУЧЕНИЕ БЕЗ УЧИТЕЛЯ (ISOLATION FOREST НА МУЛЬТИФИЧАХ)
# ==============================================================================
def detect_anomalies_isolation_forest(df, contamination=0.045):
    """
    Обучает Isolation Forest на векторе сконструированных признаков:
      - Текущее значение RPS
      - Первая разность (скорость скачка: dy/dt)
      - Отклонение от типичного часа дня
      - Разность со скользящим средним за 12 часов
    """
    features = pd.DataFrame(index=df.index)
    features["rps"] = df["rps"]
    features["diff_1"] = df["rps"].diff().fillna(0.0)
    
    # Отклонение от часового профиля
    hourly_avg = df.groupby(df.index.hour)["rps"].transform("median")
    features["dev_from_hourly"] = df["rps"] - hourly_avg

    # Отклонение от локального скользящего среднего
    features["dev_from_rolling12"] = df["rps"] - df["rps"].rolling(12, min_periods=1).mean()

    iso_model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42
    )
    pred_labels = iso_model.fit_predict(features)
    # В sklearn: -1 = аномалия, 1 = норма
    is_anomaly = (pred_labels == -1).astype(int)
    
    return pd.Series(is_anomaly, index=df.index)


# ==============================================================================
# 5. ОСНОВНОЙ ПАЙПЛАЙН: ОБНАРУЖЕНИЕ, ОЦЕНКА МЕТРИК И СРАВНЕНИЕ
# ==============================================================================
def run_anomaly_detection_pipeline():
    print("=" * 70)
    print("🔍 3. ДЕТЕКЦИЯ АНОМАЛИЙ ВО ВРЕМЕННЫХ РЯДАХ (SERVER METRICS)")
    print("=" * 70)

    # 1. Генерация ряда с аномалиями
    df = generate_server_metrics_with_anomalies(n_days=30, random_seed=42)
    y_true = df["is_anomaly"].values
    n_total_anomalies = np.sum(y_true)
    print(f"✓ Сгенерирован ряд мониторинга: {len(df)} точек (30 дней)")
    print(f"✓ Внедрено истинных аномалий: {n_total_anomalies} точек ({n_total_anomalies/len(df)*100:.1f}%)")
    print(f"   - Точечные выбросы (Spikes): 3 шт.")
    print(f"   - Контекстная аномалия (Ночной всплеск): 3 точки")
    print(f"   - Сдвиг уровня (Level Shift): 25 точек")

    # 2. Запуск Метода 1: Seasonal Rolling MAD
    print("\n[Метод 1] Сканирование сезонно-скорректированным коридором (Seasonal MAD)...")
    pred_mad, lower_mad, upper_mad = detect_anomalies_seasonal_mad(df["rps"], threshold_sigma=3.0)
    p_mad = precision_score(y_true, pred_mad, zero_division=0)
    r_mad = recall_score(y_true, pred_mad, zero_division=0)
    f1_mad = f1_score(y_true, pred_mad, zero_division=0)

    # 3. Запуск Метода 2: Декомпозиция STL Residuals
    print("[Метод 2] Сканирование остатков суточной декомпозиции (STL Residuals)...")
    pred_stl, resid_stl, resid_threshold = detect_anomalies_stl(df["rps"], period=24, threshold_sigma=2.8)
    p_stl = precision_score(y_true, pred_stl, zero_division=0)
    r_stl = recall_score(y_true, pred_stl, zero_division=0)
    f1_stl = f1_score(y_true, pred_stl, zero_division=0)

    # 4. Запуск Метода 3: Isolation Forest
    print("[Метод 3] Обучение многомерного детектора Isolation Forest...")
    contamination_rate = n_total_anomalies / len(df)
    pred_iso = detect_anomalies_isolation_forest(df, contamination=contamination_rate)
    p_iso = precision_score(y_true, pred_iso, zero_division=0)
    r_iso = recall_score(y_true, pred_iso, zero_division=0)
    f1_iso = f1_score(y_true, pred_iso, zero_division=0)

    # 5. Сводная таблица метрик
    print("\n" + "=" * 65)
    print(f"{'Алгоритм обнаружения':<28} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 65)
    print(f"{'1. Seasonal Rolling MAD':<28} | {p_mad*100:6.1f}%   | {r_mad*100:6.1f}%   | {f1_mad*100:6.1f}%")
    print(f"{'2. STL Residual Analysis':<28} | {p_stl*100:6.1f}%   | {r_stl*100:6.1f}%   | {f1_stl*100:6.1f}%")
    print(f"{'3. Isolation Forest ML':<28}   | {p_iso*100:6.1f}%   | {r_iso*100:6.1f}%   | {f1_iso*100:6.1f}%")
    print("=" * 65)

    print("\n💡 ЭКСПЕРТНЫЕ ВЫВОДЫ:")
    print("• Seasonal MAD формирует динамический коридор, следующий за суточным циклом, отлично фиксируя сдвиги уровней и пики.")
    print("• STL Residual Analysis блестяще изолирует скрытые ночные всплески благодаря вычитанию суточного профиля.")
    print("• Isolation Forest оценивает точки комплексно в многомерном пространстве (величина + дифференциал dy/dt).")

    # 6. Построение наглядного аналитического графика
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Панель 1: Исходный ряд + коридор нормы MAD + Ground Truth аномалии
    axes[0].plot(df.index, df["rps"], color="#34495e", lw=1.2, label="Сетевой трафик (RPS)")
    axes[0].fill_between(df.index, lower_mad, upper_mad, color="#2ecc71", alpha=0.25, label="Сезонный динамический коридор нормы (MAD ±3σ)")
    
    # Подсветка истинных аномалий
    true_anom_df = df[df["is_anomaly"] == 1]
    axes[0].scatter(true_anom_df.index, true_anom_df["rps"], color="#e74c3c", s=45, zorder=5, label="Истинные аномалии (Ground Truth)", edgecolors="black")
    axes[0].set_title("1. Исходный ряд мониторинга серверов и адаптивный коридор нормы (Seasonal MAD)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("RPS", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    # Панель 2: Анализ остатков STL декомпозиции
    axes[1].plot(df.index, resid_stl, color="#8e44ad", lw=1.2, label="Остаточный шум декомпозиции (Residual)")
    axes[1].axhline(y=resid_threshold, color="red", linestyle="--", label=f"Порог аномалии (±{resid_threshold:.1f})")
    axes[1].axhline(y=-resid_threshold, color="red", linestyle="--")
    
    # Точки, превысившие порог STL
    stl_flagged = df[pred_stl == 1]
    axes[1].scatter(stl_flagged.index, resid_stl.loc[stl_flagged.index], color="#e74c3c", marker="x", s=50, label="Детектировано STL", zorder=5)
    axes[1].set_title("2. Метод STL: Очищенный остаточный сигнал (Идеально выделяет скрытый ночной пик и импульсы)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Ошибка остатка", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    # Панель 3: Сравнение детекции трех методов
    axes[2].plot(df.index, df["rps"], color="#bdc3c7", lw=1.0, alpha=0.7)
    
    # Маркеры детекции каждым методом
    mad_hits = df[pred_mad == 1]
    stl_hits = df[pred_stl == 1]
    iso_hits = df[pred_iso == 1]

    axes[2].scatter(mad_hits.index, mad_hits["rps"], color="#3498db", marker="^", s=65, label=f"Seasonal MAD (F1={f1_mad*100:.1f}%)", zorder=4)
    axes[2].scatter(stl_hits.index, stl_hits["rps"], color="#9b59b6", marker="s", s=45, label=f"STL Residuals (F1={f1_stl*100:.1f}%)", zorder=5)
    axes[2].scatter(iso_hits.index, iso_hits["rps"], color="#e67e22", marker="o", s=30, label=f"Isolation Forest (F1={f1_iso*100:.1f}%)", zorder=6)

    axes[2].set_title("3. Сопоставление срабатываний всех 3 алгоритмов", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("RPS", fontsize=11)
    axes[2].set_xlabel("Временная шкала (Дата)", fontsize=11)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plot_path = os.path.join("plots", "03_anomaly_detection.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n📊 График сохранен в файл: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_anomaly_detection_pipeline()
