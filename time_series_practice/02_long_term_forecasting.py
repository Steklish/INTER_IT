"""
================================================================================
ПРИМЕР 2: ДОЛГОСРОЧНОЕ ПРОГНОЗИРОВАНИЕ ВРЕМЕННЫХ РЯДОВ (LONG-TERM FORECASTING)
================================================================================

В этом примере мы решаем задачу прогнозирования на длинный горизонт:
  - Прогноз ежедневных продаж/спроса на 90 дней (3 месяца) вперед.
  - Типично для: планирования закупок на склад, составления годового бюджета компании,
    прогнозирования мощности серверов на квартал.

ПОЧЕМУ КРАТКОСРОЧНЫЙ ПОДХОД ЗДЕСЬ ПРОВАЛИВАЕТСЯ:
Если мы попытаемся использовать лаги вчерашнего дня (lag_1, lag_2), то на шаге t+45
мы НЕ знаем реального вчерашнего дня (t+44).
Если применять рекурсивный прогноз на 90 шагов вперед, малейшая неточность на шаге 1
будет накапливаться, и к 30-му шагу прогноз выродится в прямую линию или улетит
в бесконечность (Compounding Error Drift).

КАК ПРАВИЛЬНО ДЕЛАТЬ ДОЛГОСРОЧНЫЙ ПРОГНОЗ:
1. Декомпозиция ряда (STL / Classical Decomposition):
   Ряд раскладывается на три слагаемых:
      Y(t) = Trend(t) + Seasonality(t) + Residual(t)
   - Trend: общий вектор развития (бизнес растет на 15% в год).
   - Seasonality: стабильные циклические колебания (годовые и недельные паттерны).
   - Residual: случайный шум, который нельзя предсказать на 3 месяца вперед.

2. Детерминированные признаки будущего и гармоники Фурье (Fourier Terms):
   Мы обучаем модель ML только на тех признаках, которые ТОЧНО ИЗВЕСТНЫ в будущем:
   - Гармоники Фурье (sin/cos от дня года): описывают плавную годовую волну
   - Календарные маркеры: день недели, месяц, праздничные всплески
   - Долгие сезонные лаги: продажи ровно 1 год назад (lag_365), которые УЖЕ есть в истории!

3. Доверительный интервал (Uncertainty Intervals):
   Чем дальше в будущее мы прогнозируем, тем шире становится коридор неопределенности.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
import lightgbm as lgb


# ==============================================================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ ЕЖЕДНЕВНЫХ ПРОДАЖ (3 ГОДА ИСТОРИИ)
# ==============================================================================
def generate_retail_sales_data(n_days=1095, random_seed=42):
    """
    Генерирует ежедневные продажи интернет-магазина за 3 года (1095 дней):
      - Долгосрочный тренд роста бизнеса (+20% в год)
      - Годовая сезонность (пик перед Новым Годом в декабре, спад в феврале)
      - Недельная сезонность (пик заказов в пятницу и субботу)
      - Распродажи Черной Пятницы (Black Friday)
      - Случайный шум
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    # 1. Тренд: базовый уровень 1000 заказов + квадратичный рост
    trend = 1000.0 + 0.5 * t + 0.0003 * (t ** 2)

    # 2. Годовая сезонность (день в году от 1 до 365)
    day_of_year = timestamps.dayofyear.values
    yearly_seasonality = (
        180.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25) +
        90.0 * np.cos(4 * np.pi * (day_of_year - 330) / 365.25)
    )

    # 3. Недельная сезонность (день недели от 0=Пн до 6=Вс)
    day_of_week = timestamps.dayofweek.values
    weekly_factors = np.array([-40, -20, 0, 30, 80, 100, -30])
    weekly_seasonality = weekly_factors[day_of_week]

    # 4. Всплеск Черной пятницы (конец ноября)
    black_friday_spike = np.zeros(n_days)
    for year in [2023, 2024, 2025]:
        bf_mask = (timestamps.year == year) & (timestamps.month == 11) & (timestamps.day >= 24) & (timestamps.day <= 27)
        black_friday_spike[bf_mask] = 450.0

    # 5. Случайный шум
    noise = np.random.normal(loc=0.0, scale=45.0, size=n_days)

    sales = trend + yearly_seasonality + weekly_seasonality + black_friday_spike + noise
    sales = np.clip(sales, a_min=100.0, a_max=None)

    df = pd.DataFrame({
        "date": timestamps,
        "sales": sales
    }).set_index("date")

    return df


# ==============================================================================
# 2. ПОДГОТОВКА ДЕТЕРМИНИРОВАННЫХ ПРИЗНАКОВ ДЛЯ ДОЛГОСРОЧНОГО ПРОГНОЗА
# ==============================================================================
def create_long_term_features(df, target_col="sales"):
    """
    Генерирует признаки, которые гарантированно известны на любой горизонт вперед:
      - Гармоники Фурье для моделирования годовой сезонности без ступенек
      - Календарные циклы (день недели, месяц, квартал)
      - Глобальный линейный тренд времени
      - Годовой лаг (lag_365)
    """
    data = df.copy()

    # Порядковый номер дня от начала отсчета (для тренда)
    data["time_idx"] = (data.index - data.index[0]).days

    # Календарные компоненты
    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    data["quarter"] = data.index.quarter
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # Гармоники Фурье (Fourier Terms) для годового цикла
    # С их помощью модель описывает плавную синусоидальную волну сезонности
    day_of_year = data.index.dayofyear
    for k in [1, 2, 3]:
        data[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        data[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    # Годовой лаг (продажи ровно 365 дней назад)
    data["lag_365"] = data[target_col].shift(365)

    # Доступен ли лаг? Первые 365 дней отбрасываем
    data_clean = data.dropna(subset=["lag_365"]).copy()
    return data_clean


# ==============================================================================
# 3. ОСНОВНОЙ ПАЙПЛАЙН: ДЕКОМПОЗИЦИЯ, МОДЕЛИРОВАНИЕ И ДОВЕРИТЕЛЬНЫЙ КОРИДОР
# ==============================================================================
def run_long_term_pipeline():
    print("=" * 70)
    print("📈 2. ДОЛГОСРОЧНЫЙ ПРОГНОЗ СПРОСА (ГОРИЗОНТ: 90 ДНЕЙ ВПЕРЕД)")
    print("=" * 70)

    # 1. Генерация истории продаж за 3 года
    df_raw = generate_retail_sales_data(n_days=1095, random_seed=42)
    print(f"✓ Загружен исторический ряд: {len(df_raw)} дней ({df_raw.index.min().date()} — {df_raw.index.max().date()})")

    # 2. Выделение тестового периода (последние 90 дней = 1 квартал)
    horizon_days = 90
    train_raw = df_raw.iloc[:-horizon_days]
    test_raw = df_raw.iloc[-horizon_days:]
    print(f"✓ Обучающая выборка (Train): {len(train_raw)} дней (до {train_raw.index.max().date()})")
    print(f"✓ Тестовый период (Test):    {len(test_raw)} дней ({test_raw.index.min().date()} — {test_raw.index.max().date()})")

    # 3. МЕТОД 1: Классическая декомпозиция ряда (STL / Seasonal Decompose)
    print("\n[Метод 1] Выполняем декомпозицию ряда на Тренд, Сезонность и Остаток...")
    decomp = seasonal_decompose(train_raw["sales"], model="additive", period=7)
    trend_clean = decomp.trend.dropna()
    
    # Экстраполяция тренда линейной регрессией
    poly_fit = np.polyfit(np.arange(len(trend_clean)), trend_clean.values, deg=1)
    future_trend_steps = np.arange(len(trend_clean), len(trend_clean) + horizon_days)
    extrapolated_trend = np.polyval(poly_fit, future_trend_steps)
    
    # Сезонный недельный профиль повторяем циклически
    seasonal_pattern = decomp.seasonal.iloc[-7:].values
    extrapolated_seasonal = [seasonal_pattern[i % 7] for i in range(horizon_days)]
    stl_forecast = extrapolated_trend + extrapolated_seasonal

    # 4. МЕТОД 2: Машинное обучение с гармониками Фурье и годовым лагом
    print("[Метод 2] Обучаем модель LightGBM на детерминированных признаках будущего...")
    df_features = create_long_term_features(df_raw, target_col="sales")
    feature_cols = [c for c in df_features.columns if c != "sales"]

    train_feat = df_features.loc[df_features.index.isin(train_raw.index)]
    test_feat = df_features.loc[df_features.index.isin(test_raw.index)]

    X_train, y_train = train_feat[feature_cols], train_feat["sales"]
    X_test, y_test = test_feat[feature_cols], test_feat["sales"]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=20,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)

    y_pred_lgb = lgb_model.predict(X_test)

    # 5. Оценка доверительного коридора (Uncertainty Band 90%)
    # Стандартное отклонение ошибок на обучении
    train_residuals = y_train - lgb_model.predict(X_train)
    residual_std = np.std(train_residuals)
    
    # Доверительный коридор расширяется по мере удаления в будущее
    time_expansion = np.linspace(1.0, 1.35, horizon_days)
    z_score_90 = 1.645  # 90% интервал для нормального распределения
    lower_bound = y_pred_lgb - z_score_90 * residual_std * time_expansion
    upper_bound = y_pred_lgb + z_score_90 * residual_std * time_expansion

    # 6. Сравнение метрик качества на горизонте 90 дней
    # Бейзлайн: продажи ровно год назад (Seasonal Naive 365)
    y_naive_365 = X_test["lag_365"].values

    mae_base = mean_absolute_error(y_test, y_naive_365)
    mape_base = mean_absolute_percentage_error(y_test, y_naive_365) * 100

    mae_stl = mean_absolute_error(y_test, stl_forecast)
    mape_stl = mean_absolute_percentage_error(y_test, stl_forecast) * 100

    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    mape_lgb = mean_absolute_percentage_error(y_test, y_pred_lgb) * 100

    # Процент покрытия доверительным интервалом
    coverage_90 = np.mean((y_test.values >= lower_bound) & (y_test.values <= upper_bound)) * 100

    print("\n--- РЕЗУЛЬТАТЫ ДОЛГОСРОЧНОГО ПРОГНОЗА (90 ДНЕЙ ТЕСТА) ---")
    print(f"1. Бейзлайн (Ровно год назад, lag_365):  MAE = {mae_base:.2f} | MAPE = {mape_base:.2f}%")
    print(f"2. Декомпозиция STL (Тренд + Неделя):     MAE = {mae_stl:.2f} | MAPE = {mape_stl:.2f}%")
    print(f"3. LightGBM (Фурье + Календарь + lag_365): MAE = {mae_lgb:.2f} | RMSE = {rmse_lgb:.2f} | MAPE = {mape_lgb:.2f}%")
    print(f"🏆 Превосходство LightGBM над наивным прошлым годом: ошибка ниже на {((mae_base - mae_lgb)/mae_base)*100:.1f}%!")
    print(f"🛡️ Фактическое покрытие 90% коридора неопределенности: {coverage_90:.1f}%")

    # 7. Визуализация и построение графиков
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Верхний график: История и прогноз на 90 дней вперед с доверительным коридором
    # Показываем последние 180 дней истории + 90 дней теста
    history_view = train_raw.iloc[-180:]
    axes[0].plot(history_view.index, history_view["sales"], label="История продаж (Train)", color="#7f8c8d", lw=1.5)
    axes[0].plot(test_raw.index, test_raw["sales"], label="Фактические продажи (Ground Truth)", color="black", lw=2.2)
    axes[0].plot(test_raw.index, stl_forecast, label=f"Прогноз STL [MAPE={mape_stl:.1f}%]", color="#9b59b6", ls="--", lw=1.8)
    axes[0].plot(test_raw.index, y_pred_lgb, label=f"Прогноз LightGBM [MAPE={mape_lgb:.1f}%]", color="#2980b9", lw=2.2)
    
    # Закраска 90%-го доверительного интервала
    axes[0].fill_between(
        test_raw.index, lower_bound, upper_bound,
        color="#3498db", alpha=0.22,
        label=f"90% Доверительный коридор (Покрытие {coverage_90:.1f}%)"
    )

    axes[0].axvline(x=test_raw.index[0], color="red", linestyle=":", lw=2, label="Старт долгосрочного прогноза (t_0)")
    axes[0].set_title("Долгосрочный прогноз спроса на 90 дней (Квартал вперед)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Объем продаж (заказов/день)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left", frameon=True)

    # Нижний график: Компоненты декомпозиции (Тренд и Сезонность)
    decomp_recent = seasonal_decompose(df_raw.iloc[-365:]["sales"], model="additive", period=7)
    axes[1].plot(decomp_recent.trend.index, decomp_recent.trend, label="Выделенный глобальный тренд (Trend)", color="#e74c3c", lw=2)
    axes[1].plot(decomp_recent.seasonal.index, decomp_recent.seasonal + 1200, label="Недельный паттерн со смещением (+1200)", color="#27ae60", alpha=0.7)
    axes[1].set_title("Компоненты декомпозиции ряда за последний год (Выделение тренда и паттернов)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Значение компонентов", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plot_path = os.path.join("plots", "02_long_term_forecast.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n📊 График сохранен в файл: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_long_term_pipeline()
