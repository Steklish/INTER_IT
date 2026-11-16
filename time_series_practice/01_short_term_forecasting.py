"""
================================================================================
ПРИМЕР 1: КРАТКОСРОЧНОЕ ПРОГНОЗИРОВАНИЕ ВРЕМЕННЫХ РЯДОВ (SHORT-TERM FORECASTING)
================================================================================

В этой задаче мы прогнозируем следующие шаги (следующие 1-48 часов) вперед во времени.
Это типичный сценарий для:
  - Почасового прогнозирования энергопотребления электросети
  - Нагрузки на веб-серверы и API (RPS / CPU)
  - Вызова такси или курьерской доставки на ближайшие часы

КЛЮЧЕВЫЕ КОНЦЕПЦИИ ДЛЯ НАЧИНАЮЩИХ:
1. Запрет на перемешивание (No Shuffle):
   Во временных рядах нельзя делать `train_test_split(shuffle=True)`. Иначе модель
   будет подсматривать в будущее (Data Leakage) и в продакшене потерпит фиаско.
   Обучение ВСЕГДА идет на прошлом, тестирование — на будущем!

2. Лаги (Lags) и Скользящие статистики (Rolling):
   Чтобы обучить модель табличного машинного обучения (например, LightGBM), мы превращаем
   одномерный временной ряд в табличную матрицу признаков:
   - Лаг 1: значение час назад (y_{t-1})
   - Лаг 24: значение ровно сутки назад (y_{t-24})
   - Скользящее среднее: среднее за последние 6 и 24 часа. ВАЖНО: всегда применяем
     .shift(1), иначе текущее известное значение утечет в признаки!

3. Одношаговый прогноз vs Рекурсивный многошаговый:
   - One-Step: прогнозируем только t+1 (каждый шаг знает реальное прошлое).
   - Recursive Multi-Step: прогнозируем t+1, подставляем прогноз как факт,
     прогнозируем t+2, t+3 ... t+24 (накопление ошибки).
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb


# ==============================================================================
# 1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО РЕАЛИСТИЧНОГО ВРЕМЕННОГО РЯДА
# ==============================================================================
def generate_energy_demand_data(n_days=180, random_seed=42):
    """
    Генерирует почасовые данные энергопотребления (МВт) за n_days дней.
    Включает:
      - Базовый уровень потребления (~500 МВт)
      - Суточный цикл (2 пика: утро ~09:00 и вечер ~19:00, спад ночью ~04:00)
      - Недельный цикл (в субботу и воскресенье потребление на заводах ниже)
      - Сезонный плавный тренд (погода/температура)
      - Случайный белый шум и случайные локальные всплески
    """
    np.random.seed(random_seed)
    n_hours = n_days * 24
    timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=n_hours, freq="h")

    # Базовая линия
    base = 500.0

    # 1. Суточный профиль (24 часа)
    hours = timestamps.hour.values
    daily_profile = (
        120 * np.sin(2 * np.pi * (hours - 5) / 24) +
        40 * np.sin(4 * np.pi * (hours - 8) / 24)
    )

    # 2. Недельный профиль (выходные дни тише на 70 МВт)
    day_of_week = timestamps.dayofweek.values
    weekend_mask = (day_of_week >= 5).astype(float)
    weekly_profile = -70.0 * weekend_mask

    # 3. Долгосрочный медленный тренд (зима -> весна -> лето)
    t = np.linspace(0, 3 * np.pi, n_hours)
    seasonal_trend = 50.0 * np.sin(t)

    # 4. Случайный случайный шум
    noise = np.random.normal(loc=0.0, scale=18.0, size=n_hours)

    consumption = base + daily_profile + weekly_profile + seasonal_trend + noise
    consumption = np.clip(consumption, a_min=100.0, a_max=None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "energy_mw": consumption
    }).set_index("timestamp")

    return df


# ==============================================================================
# 2. FEATURE ENGINEERING (КОНСТРУИРОВАНИЕ ПРИЗНАКОВ ДЛЯ ВРЕМЕННЫХ РЯДОВ)
# ==============================================================================
def create_time_series_features(df, target_col="energy_mw"):
    """
    Превращает временной ряд в матрицу признаков:
      - Календарные и циклические гармоники (Hour sin/cos, DayOfWeek)
      - Лаги (значения в прошлом)
      - Скользящие средние и отклонения
    """
    data = df.copy()

    # 1. Календарные признаки
    data["hour"] = data.index.hour
    data["dayofweek"] = data.index.dayofweek
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # Циклическое кодирование часов (23:00 и 00:00 близки геометрически!)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)

    # 2. Лаги (значения из прошлого)
    # lag_1: час назад
    # lag_2: 2 часа назад
    # lag_24: ровно сутки назад (очень сильный признак для суточных рядов!)
    # lag_168: ровно неделю назад (тот же час в тот же день недели)
    for lag in [1, 2, 3, 24, 48, 168]:
        data[f"lag_{lag}"] = data[target_col].shift(lag)

    # 3. Скользящие статистики (Rolling Features)
    # КРИТИЧЕСКИЙ МОМЕНТ: Мы берем shift(1) перед rolling!
    # Иначе в среднее попадет само целевое значение часа t, что недопустимо.
    shifted = data[target_col].shift(1)
    data["rolling_mean_6h"] = shifted.rolling(window=6).mean()
    data["rolling_std_6h"] = shifted.rolling(window=6).std()
    data["rolling_mean_24h"] = shifted.rolling(window=24).mean()
    data["rolling_std_24h"] = shifted.rolling(window=24).std()

    # Удаляем строки, где лаги еще не накопились (первые 168 часов)
    # ВАЖНО: проверяем только признаки, а не target_col, чтобы можно было генерировать фичи для будущих дат!
    feature_columns = [c for c in data.columns if c != target_col]
    data_clean = data.dropna(subset=feature_columns).copy()
    return data_clean


# ==============================================================================
# 3. РЕКУРСИВНЫЙ МНОГОШАГОВЫЙ ПРОГНОЗ (RECURSIVE MULTI-STEP)
# ==============================================================================
def recursive_forecast(model, history_df, steps=24, feature_cols=None, target_col="energy_mw"):
    """
    Рекурсивно прогнозирует `steps` шагов вперед:
      1. На шаге 1 использует реальную историю для вычисления лагов.
      2. Предсказывает y_{t+1}.
      3. Добавляет y_{t+1} в историю и пересчитывает лаги для t+2.
      4. Повторяет до шага `steps`.
    """
    current_history = history_df.copy()
    predictions = []

    last_timestamp = current_history.index[-1]
    future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=steps, freq="h")

    for step_ts in future_timestamps:
        # 1. Создаем признаки для текущей точки на основе имеющейся истории (включая прошлые предсказания)
        # Добавляем фиктивную строку для step_ts
        temp_df = pd.concat([current_history, pd.DataFrame({target_col: [np.nan]}, index=[step_ts])])
        feat_df = create_time_series_features(temp_df, target_col=target_col)

        row_features = feat_df.loc[[step_ts], feature_cols]
        pred_val = model.predict(row_features)[0]

        predictions.append(pred_val)
        # Записываем прогноз в историю для следующих шагов
        current_history.loc[step_ts, target_col] = pred_val

    return pd.Series(predictions, index=future_timestamps, name="pred_recursive")


# ==============================================================================
# 4. ОСНОВНОЙ ПАЙПЛАЙН: ОБУЧЕНИЕ, ВАЛИДАЦИЯ И ВИЗУАЛИЗАЦИЯ
# ==============================================================================
def run_short_term_pipeline():
    print("=" * 70)
    print("🚀 1. КРАТКОСРОЧНЫЙ ПРОГНОЗ: ПОЧАСОВОЕ ЭНЕРГОПОТРЕБЛЕНИЕ")
    print("=" * 70)

    # 1. Загрузка/генерация данных
    df_raw = generate_energy_demand_data(n_days=180, random_seed=42)
    print(f"✓ Сгенерирован временной ряд: {len(df_raw)} часов ({df_raw.index.min().date()} — {df_raw.index.max().date()})")

    # 2. Конструирование признаков
    df_features = create_time_series_features(df_raw, target_col="energy_mw")
    feature_cols = [c for c in df_features.columns if c != "energy_mw"]
    print(f"✓ Сконструировано {len(feature_cols)} признаков (лаги, скользящие окна, гармоники времени)")

    # 3. Хронологическое разделение на Train и Test (ПОСЛЕДНИЕ 14 ДНЕЙ НА ТЕСТ)
    test_hours = 14 * 24
    train_df = df_features.iloc[:-test_hours]
    test_df = df_features.iloc[-test_hours:]

    X_train, y_train = train_df[feature_cols], train_df["energy_mw"]
    X_test, y_test = test_df[feature_cols], test_df["energy_mw"]

    print(f"✓ Train: {len(X_train)} часов ({train_df.index.min().date()} — {train_df.index.max().date()})")
    print(f"✓ Test:  {len(X_test)} часов ({test_df.index.min().date()} — {test_df.index.max().date()})")

    # 4. Обучение модели LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    print("✓ Модель LightGBM успешно обучена!")

    # 5. Оценка One-Step прогноза на всем тестовом периоде (14 дней)
    y_pred_one_step = model.predict(X_test)

    # Бейзлайн: Seasonal Naive (прогноз равен значению ровно 24 часа назад)
    y_baseline_naive = X_test["lag_24"].values

    mae_base = mean_absolute_error(y_test, y_baseline_naive)
    mape_base = mean_absolute_percentage_error(y_test, y_baseline_naive) * 100

    mae_lgb = mean_absolute_error(y_test, y_pred_one_step)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_one_step))
    mape_lgb = mean_absolute_percentage_error(y_test, y_pred_one_step) * 100

    print("\n--- РЕЗУЛЬТАТЫ СКОРИНГА (ONE-STEP НА 14 ДНЕЙ ТЕСТА) ---")
    print(f"Бейзлайн (Seasonal Naive, lag_24):  MAE = {mae_base:.2f} МВт | MAPE = {mape_base:.2f}%")
    print(f"Модель LightGBM (One-Step ahead):   MAE = {mae_lgb:.2f} МВт | RMSE = {rmse_lgb:.2f} МВт | MAPE = {mape_lgb:.2f}%")
    print(f"🏆 Превосходство LightGBM над бейзлайном: ошибка снижена на {((mae_base - mae_lgb) / mae_base)*100:.1f}%!")

    # 6. Рекурсивный прогноз на ближайшие 48 часов вперед
    horizon_hours = 48
    history_for_recursive = df_raw.loc[:train_df.index[-1], ["energy_mw"]]
    print(f"\nВыполняем рекурсивный прогноз на {horizon_hours} часов вперед (авторегрессия без истинных данных)...")
    recursive_preds = recursive_forecast(
        model=model,
        history_df=history_for_recursive,
        steps=horizon_hours,
        feature_cols=feature_cols,
        target_col="energy_mw"
    )

    actual_48h = test_df.iloc[:horizon_hours]["energy_mw"]
    mae_rec = mean_absolute_error(actual_48h, recursive_preds)
    mape_rec = mean_absolute_percentage_error(actual_48h, recursive_preds) * 100
    print(f"Рекурсивный прогноз на 48 часов: MAE = {mae_rec:.2f} МВт | MAPE = {mape_rec:.2f}%")

    # 7. Визуализация и сохранение графиков
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Верхний график: 48-часовой срез (Факт vs Бейзлайн vs One-Step vs Recursive)
    slice_48h = test_df.iloc[:horizon_hours]
    axes[0].plot(slice_48h.index, slice_48h["energy_mw"], label="Фактическое потребление (True)", color="black", lw=2.5)
    axes[0].plot(slice_48h.index, slice_48h["lag_24"], label=f"Бейзлайн (Сутки назад) [MAPE={mape_base:.1f}%]", color="gray", ls="--", alpha=0.7)
    axes[0].plot(slice_48h.index, y_pred_one_step[:horizon_hours], label=f"LightGBM One-Step [MAPE={mape_lgb:.1f}%]", color="#007acc", lw=2)
    axes[0].plot(recursive_preds.index, recursive_preds.values, label=f"LightGBM Recursive 48h [MAPE={mape_rec:.1f}%]", color="#e67e22", lw=2, ls="-.")
    
    axes[0].set_title("Краткосрочный прогноз энергопотребления (Срез на 48 часов)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Потребление энергии (МВт)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    # Нижний график: Важность признаков (Feature Importance)
    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    top_feats = feat_imp.tail(10)
    top_feats.plot(kind="barh", ax=axes[1], color="#2ecc71", edgecolor="black")
    axes[1].set_title("Топ-10 самых информативных признаков (LightGBM Importance)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Количество сплитов признака в деревьях", fontsize=11)
    axes[1].grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join("plots", "01_short_term_forecast.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n📊 График сохранен в файл: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_short_term_pipeline()
