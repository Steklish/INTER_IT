"""
===============================================================================
TUTORIAL 07: PRODUCTION ML PIPELINE (Сквозной промышленный пайплайн)
===============================================================================

1. ЧТО ЭТО И ЗАЧЕМ НУЖНО?
------------------------
В реальном мире данные НИКОГДА не приходят чистыми, числовыми и аккуратными:
  - Есть пропуски (NaN, None, пустые ячейки).
  - Есть текст и категории ('город', 'тип тарифа', 'марка телефона').
  - Числа имеют совершенно разный масштаб (зарплата 150 000 vs возраст 28).

Большинство новичков делают предобработку «вручную» в Pandas до разделения на
Train и Test. Это ГЛАВНЫЙ ГРЕХ Data Science — УТЕЧКА ДАННЫХ (Data Leakage)!
Информация из тестового набора просачивается в модель, и в продакшене система ломается.

2. КАК ДЕЛАЕТСЯ ПРАВИЛЬНО?
---------------------------
Используется связка:
  1. ColumnTransformer: распределяет колонки по типам (числа в одну сторону, текст в другую).
  2. Pipeline: упаковывает заполнение пропусков (Imputer), масштабирование (Scaler),
     кодирование категорий (OneHotEncoder) и саму модель в ЕДИНЫЙ объект.

3. ГЛАВНОЕ ПРЕИМУЩЕСТВО ДЛЯ ПРОДАКШЕНА:
---------------------------------------
Мы сохраняем на диск ОДИН файл (`pipeline.joblib`).
Когда в веб-сервис приходит «сырой» JSON от пользователя:
    `model.predict(raw_json)`
Пайплайн сам заполнит пропуски, сам закодирует текст, сам смасштабирует числа
и выдаст ответ! Никакого ручного дублирования кода предобработки.
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Компоненты для сборки пайплайна
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = Path(__file__).resolve().parent / "production_pipeline.joblib"


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ «ГРЯЗНОГО» РЕАЛИСТИЧНОГО ДАТАСЕТА (ПОКУПКА КУРСА В IT-ШКОЛЕ)
# =============================================================================
print("--> Шаг 1: Создание сырых 'грязных' данных (с пропусками и категориями)")

np.random.seed(42)
n_samples = 800

# Категориальные признаки (текст):
cities = np.random.choice(['Moscow', 'Saint Petersburg', 'Novosibirsk', 'Kazan', 'Erevan'], size=n_samples)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], p=[0.2, 0.5, 0.25, 0.05], size=n_samples)
device = np.random.choice(['iOS', 'Android', 'Windows', 'MacOS'], size=n_samples)

# Числовые признаки:
age = np.random.uniform(18, 60, size=n_samples)
web_pages_viewed = np.random.poisson(lam=8, size=n_samples)
days_since_reg = np.random.uniform(1, 180, size=n_samples)

# Целевая метка: Купил ли курс (0 или 1)
prob = (
    0.2 +
    0.25 * (web_pages_viewed > 10) +
    0.20 * np.isin(device, ['iOS', 'MacOS']) +
    0.15 * np.isin(education, ['Master', 'PhD']) +
    np.random.normal(0, 0.15, size=n_samples)
)
bought_course = (prob >= 0.5).astype(int)

# Собираем сырой датафрейм:
df = pd.DataFrame({
    'city': cities,
    'education': education,
    'device': device,
    'age': np.round(age, 1),
    'web_pages_viewed': web_pages_viewed,
    'days_since_reg': np.round(days_since_reg, 1),
    'bought_course': bought_course
})

# НАМЕРЕННО ДОБАВЛЯЕМ ПРОПУСКИ (NaN) КАК В ЖИЗНИ:
# В 8% случаев возраст неизвестен
df.loc[np.random.rand(n_samples) < 0.08, 'age'] = np.nan
# В 5% случаев город не указан
df.loc[np.random.rand(n_samples) < 0.05, 'city'] = np.nan

print("\nПервые 5 строк 'сырого' датасета:")
print(df.head())
print(f"\nКоличество пропусков (NaN) в колонках:")
print(df.isna().sum())


# =============================================================================
# ШАГ 2: РАЗДЕЛЕНИЕ НА X, y И НА TRAIN / TEST
# =============================================================================
print("\n--> Шаг 2: Разделение данных ДО любых манипуляций")

X = df.drop(columns=['bought_course'])
y = df['bought_course']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Размер Train: {len(X_train)} строк | Test: {len(X_test)} строк")


# =============================================================================
# ШАГ 3: ПРОЕКТИРОВАНИЕ ПАЙПЛАЙНА (ColumnTransformer + Pipeline)
# =============================================================================
print("\n--> Шаг 3: Конструирование промышленного пайплайна")

# Выделяем списки колонок по их физическому типу:
numeric_features = ['age', 'web_pages_viewed', 'days_since_reg']
categorical_features = ['city', 'education', 'device']

# 1. Пайплайн для чисел:
# Шаг А: Заполняем пропуски медианой (SimpleImputer)
# Шаг Б: Нормализуем масштаб (StandardScaler)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Пайплайн для категорий (текста):
# Шаг А: Заполняем пропуски словом 'missing'
# Шаг Б: Превращаем категории в 0 и 1 (One-Hot Encoding).
# handle_unknown='ignore' защищает модель: если в будущем придет новый город, код не упадет!
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 3. Объединяем обработчики колонок:
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. ИТОГОВЫЙ СКВОЗНОЙ ПАЙПЛАЙН:
# Предобработка -> Модель машинного обучения
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
])

print("Архитектура пайплайна:")
print(full_pipeline)


# =============================================================================
# ШАГ 4: КРОСС-ВАЛИДАЦИЯ (5-Fold Cross Validation)
# =============================================================================
print("\n--> Шаг 4: Надежная оценка качества через 5-Fold Cross Validation")
# Данные делятся на 5 частей: на 4 частях обучается, на 5-й проверяется (и так 5 раз)
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='accuracy')

print(f"Точность на фолдах: {[round(s, 3) for s in cv_scores]}")
print(f"Средняя точность на валидации: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")


# =============================================================================
# ШАГ 5: ФИНАЛЬНОЕ ОБУЧЕНИЕ И СОХРАНЕНИЕ В ФАЙЛ
# =============================================================================
print("\n--> Шаг 5: Обучение на всем Train и сохранение в .joblib")

# Одной строчкой обучается ВСЯ цепочка: Imputer -> Scaler -> OneHot -> RandomForest!
full_pipeline.fit(X_train, y_train)

# Тестируем на отложенной выборке:
y_pred = full_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Финальная точность на Test: {test_accuracy * 100:.2f}%")

# Сохраняем готовую модель на диск:
joblib.dump(full_pipeline, MODEL_SAVE_PATH)
print(f"Пайплайн сохранен в файл: {MODEL_SAVE_PATH}")


# =============================================================================
# ШАГ 6: СИМУЛЯЦИЯ ПРОДАКШЕНА (ИНФЕРЕНС С СЫРЫМИ ДАННЫМИ)
# =============================================================================
print("\n--> Шаг 6: Симуляция работы веб-сервиса (инференс на 'сыром' клиенте)")

# Представим, что мы перезагрузили сервер и загрузили пайплайн из файла:
loaded_pipeline = joblib.load(MODEL_SAVE_PATH)

# Пришел новый клиент с сайта. Заметьте: у него ПРОПУЩЕН возраст (NaN) и город (None)!
incoming_client = pd.DataFrame([{
    'city': None,              # Пропуск!
    'education': 'Master',
    'device': 'MacOS',
    'age': np.nan,             # Пропуск!
    'web_pages_viewed': 14,
    'days_since_reg': 12.5
}])

# Передаем прямо «как есть» без всякой ручной очистки:
prediction = loaded_pipeline.predict(incoming_client)[0]
purchase_probability = loaded_pipeline.predict_proba(incoming_client)[0][1]

print("Входные сырые данные клиента (содержат NaN):")
print(incoming_client.to_dict(orient='records')[0])
print(f"\nРезультат работы пайплайна:")
print(f"Вероятность покупки курса: {purchase_probability * 100:.1f}%")
print(f"Решение системы: {'ПОКУПКА ВЫСОКОВЕРОЯТНА (Предложить скидку)' if prediction == 1 else 'Стандартная воронка'}")


# =============================================================================
# ШАГ 7: ВИЗУАЛИЗАЦИЯ (Кросс-валидация)
# =============================================================================
print("\n--> Шаг 7: Построение графика качества")

plt.figure(figsize=(8, 5))
folds = [f"Фолд {i+1}" for i in range(len(cv_scores))]
plt.bar(folds, cv_scores * 100, color='royalblue', edgecolor='black', alpha=0.8)
plt.axhline(y=cv_scores.mean() * 100, color='red', linestyle='--', label=f'Среднее: {cv_scores.mean()*100:.1f}%')
plt.ylim([50, 100])
plt.title("5-Fold Cross Validation:\nСтабильность качества на разных кусках данных")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()
plot_path = PLOTS_DIR / "07_production_pipeline.png"
plt.savefig(plot_path, dpi=120)
plt.close()

print(f"График сохранен в файл: {plot_path}")
print("\n[Поздравляем!] Вы прошли все 7 обучающих модулей.")
