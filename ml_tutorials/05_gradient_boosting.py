"""
===============================================================================
TUTORIAL 05: GRADIENT BOOSTING & LIGHTGBM (Градиентный бустинг)
===============================================================================

1. ЧТО ЭТО И ЗАЧЕМ НУЖНО?
------------------------
Градиентный бустинг (GBDT: Gradient Boosted Decision Trees) — абсолютный король
табличных данных. Именно алгоритмы семейства бустинга (LightGBM, CatBoost, XGBoost)
выигрывают более 80% всех соревнований на Kaggle по анализу таблиц и применяются
в продакшене крупнейших IT-компаний (поиск, рекомендации, баннеры, скоринг).

2. В ЧЕМ РАЗНИЦА: RANDOM FOREST (BAGGING) VS GRADIENT BOOSTING?
---------------------------------------------------------------
* Random Forest (Бэггинг):
  Деревья обучаются ПАРАЛЛЕЛЬНО и НЕЗАВИСИМО. Каждое голосует.
  Ошибка уменьшается за счет усреднения шума.

* Gradient Boosting (Бустинг):
  Деревья обучаются ПОСЛЕДОВАТЕЛЬНО.
  - Первое дерево делает грубое предсказание.
  - Второе дерево обучается НЕ на целевую переменную, а на ОШИБКИ (остатки) первого!
  - Третье дерево исправляет ошибки первых двух.
  - Каждое следующее дерево «шлифует» узкие места модели!

3. ПОЧЕМУ ИМЕННО LIGHTGBM?
--------------------------
LightGBM создан компанией Microsoft.
(+) В десятки раз быстрее классического градиентного бустинга.
(+) Растит деревья по листьям (Leaf-wise), что дает более высокую точность.
(+) Нативно умеет работать с пропущенными значениями (NaN).

4. ГЛАВНЫЕ ГИПЕРПАРАМЕТРЫ:
--------------------------
- learning_rate: «шаг обучения» (обычно 0.01 - 0.1). Чем меньше шаг, тем аккуратнее
  модель подгоняет веса и тем меньше риск переобучения.
- n_estimators: максимальное количество деревьев (обычно 100 - 1000).
- early_stopping: автоматическая остановка обучения, если качество на валидации
  перестало улучшаться N итераций подряд.
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Импортируем LightGBM
import lightgbm as lgb

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ (ОЦЕНКА СТОИМОСТИ ПОДЕРЖАННОГО АВТОМОБИЛЯ)
# =============================================================================
print("--> Шаг 1: Создание датасета автомобильного рынка (Auto Pricing)")

np.random.seed(42)
n_samples = 1500

# Признаки автомобиля:
# 1. Возраст авто в годах (0 - 20)
car_age = np.random.uniform(0, 20, size=n_samples)

# 2. Пробег в тыс. км (зависит от возраста + случайность)
mileage = car_age * 15 + np.random.normal(10, 20, size=n_samples)
mileage = np.clip(mileage, 1, 350)

# 3. Мощность двигателя в л.с. (70 - 450)
horsepower = np.random.uniform(70, 450, size=n_samples)

# 4. Объем двигателя в литрах (1.0 - 5.0)
engine_volume = np.clip(horsepower / 80 + np.random.normal(0, 0.3, size=n_samples), 1.0, 5.0)

# 5. Количество предыдущих владельцев (1 - 4)
owners = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05], size=n_samples)

# Сложная нелинейная цена:
# Новые мощные авто стоят дорого; после 5 лет цена падает нелинейно; каждый владелец снижает стоимость
base_price = 20000 + (horsepower ** 1.5) * 8.0
depreciation = np.exp(-0.08 * car_age)  # Экспоненциальное удешевление со временем
owner_penalty = (1 - (owners - 1) * 0.07)
noise = np.random.normal(0, 1500, size=n_samples)

price = (base_price * depreciation * owner_penalty) - (mileage * 15) + noise
price = np.clip(price, 1500, 120000)

df = pd.DataFrame({
    'car_age': np.round(car_age, 1),
    'mileage_k_km': np.round(mileage, 1),
    'horsepower': np.round(horsepower, 0),
    'engine_vol': np.round(engine_volume, 1),
    'owners': owners,
    'price_usd': np.round(price, 2)
})

print("\nПервые 5 автомобилей:")
print(df.head())


# =============================================================================
# ШАГ 2: РАЗДЕЛЕНИЕ НА TRAIN, VALIDATION И TEST
# =============================================================================
print("\n--> Шаг 2: Разделение данных на 3 выборки (Train, Val, Test)")

# В бустинге нам нужна валидационная выборка (Validation Set) для ранней остановки (Early Stopping)!
X = df.drop(columns=['price_usd'])
y = df['price_usd']

# Сначала отделяем финальный Test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Оставшиеся 85% делим на Train (70% от всех) и Validation (15% от всех)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

print(f"Обучающая выборка (Train):     {len(X_train)} авто")
print(f"Валидационная выборка (Val):   {len(X_val)} авто (для контроля переобучения)")
print(f"Финальный тест (Test):         {len(X_test)} авто")


# =============================================================================
# ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ С РАННЕЙ ОСТАНОВКОЙ (EARLY STOPPING)
# =============================================================================
print("\n--> Шаг 3: Обучение LightGBM Regressor")

# Создаем модель:
model = lgb.LGBMRegressor(
    n_estimators=500,        # Максимум 500 деревьев
    learning_rate=0.05,      # Плавный шаг обучения
    num_leaves=31,           # Количество листьев в одном дереве
    random_state=42,
    verbose=-1               # Отключаем лишний технический спам
)

# Обучаем, передавая eval_set для мониторинга ошибки на каждой итерации
# early_stopping(stopping_rounds=30) остановит обучение, если ошибка на валидации
# не улучшалась 30 деревьев подряд!
callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=False),
    lgb.record_evaluation(eval_result := {})
]

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['train', 'val'],
    callbacks=callbacks
)

print(f"Обучение завершено!")
print(f"Лучшая итерация (число деревьев): {model.best_iteration_} из 500 возможных")


# =============================================================================
# ШАГ 4: ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ
# =============================================================================
print("\n--> Шаг 4: Тестирование качества")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Средняя ошибка (MAE):   ${mae:,.2f}")
print(f"Корень из MSE (RMSE):   ${rmse:,.2f}")
print(f"Коэффициент R²:         {r2:.4f} (Великолепное соответствие!)")


# =============================================================================
# ШАГ 5: ВАЖНОСТЬ ПРИЗНАКОВ
# =============================================================================
print("\n--> Шаг 5: Важность факторов по версии LightGBM")
feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
for col, val in feature_imp.items():
    print(f"  {col:15s}: {val} разбиений (splits)")


# =============================================================================
# ШАГ 6: ИНФЕРЕНС (Оценка стоимости конкретной машины)
# =============================================================================
print("\n--> Шаг 6: Оценка автомобиля клиента")

car = pd.DataFrame([{
    'car_age': 4.5,
    'mileage_k_km': 65.0,
    'horsepower': 249.0,
    'engine_vol': 2.0,
    'owners': 1
}])

predicted_val = model.predict(car)[0]
print("Автомобиль: 4.5 года, 65k км пробег, 249 л.с., 2.0L, 1 владелец")
print(f"Рекомендованная цена продажи: ${predicted_val:,.2f}")


# =============================================================================
# ШАГ 7: ВИЗУАЛИЗАЦИЯ (Кривая обучения + Предсказания)
# =============================================================================
print("\n--> Шаг 7: Построение графиков")

plt.figure(figsize=(14, 5))

# График 1: Кривая обучения (Learning Curve)
plt.subplot(1, 2, 1)
train_rmse = eval_result['train']['l2']
val_rmse = eval_result['val']['l2']
iterations = range(1, len(train_rmse) + 1)

plt.plot(iterations, np.sqrt(train_rmse), label='Train RMSE (Ошибка на обучении)', color='blue')
plt.plot(iterations, np.sqrt(val_rmse), label='Val RMSE (Ошибка на валидации)', color='red')
plt.axvline(x=model.best_iteration_, color='green', linestyle='--', label=f'Best Tree ({model.best_iteration_})')
plt.title("Процесс бустинга: Каждое новое дерево уменьшает ошибку")
plt.xlabel("Итерация бустинга (номер добавленного дерева)")
plt.ylabel("RMSE ($)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

# График 2: Реальная цена vs Предсказанная на Test
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='darkorange', alpha=0.6, edgecolors='k')
ideal_vals = np.linspace(min(y_test), max(y_test), 100)
plt.plot(ideal_vals, ideal_vals, color='black', linestyle='--', label='Идеальное совпадение')
plt.title(f"LightGBM: Реальная vs Предсказанная цена (R² = {r2:.4f})")
plt.xlabel("Реальная цена ($)")
plt.ylabel("Предсказанная моделью цена ($)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()
plot_path = PLOTS_DIR / "05_gradient_boosting.png"
plt.savefig(plot_path, dpi=120)
plt.close()

print(f"График сохранен в файл: {plot_path}")
print("\n[Готово!] Запустите следующий скрипт: 06_kmeans_clustering.py")
