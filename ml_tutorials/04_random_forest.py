"""
===============================================================================
TUTORIAL 04: RANDOM FOREST (Случайный лес)
===============================================================================

1. ЧТО ЭТО И ЗАЧЕМ НУЖНО?
------------------------
Если одно дерево решений склонно к ошибкам и переобучению, почему бы не создать
целый «лес» из сотен независимых деревьев и не устроить голосование большинства?
Это идея АНСАМБЛЕЙ (Ensemble Learning) — «мудрость толпы».

Случайный лес (Random Forest) — один из самых надежных и безотказных алгоритмов
в машинном обучении. В 90% задач на табличных данных он дает отличный результат
«из коробки», почти не требуя настройки.

2. КАК ЭТО РАБОТАЕТ ВНУТРИ? (ТЕХНИКА BAGGING)
---------------------------------------------
Чтобы деревья не получились одинаковыми (иначе толпа не имеет смысла),
алгоритм вносит случайность двумя путями:
  1. Случайные строки (Bootstrap): каждое дерево обучается на случайной
     подвыборке данных с возвращением.
  2. Случайные колонки (Feature Subsampling): в каждом узле дерево выбирает
     лучший сплит только из случайного подмножества признаков (обычно корень из N).
  3. Агрегация (Aggregation):
     - В классификации: большинство голосов (Majority Vote).
     - В регрессии: среднее арифметическое всех ответов.

3. ПЛЮСЫ И МИНУСЫ:
------------------
(+) Практически не переобучается при увеличении числа деревьев (n_estimators).
(+) Не чувствителен к масштабу признаков и выбросам.
(+) Легко параллелится на все ядра процессора (параметр n_jobs=-1).
(-) Медленнее в инференсе, чем одно дерево или линейная регрессия.
(-) «Черный ящик» — нельзя нарисовать одну понятную схему, как для одного дерева.
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ (ОТТОК СОТРУДНИКОВ ИЗ КОМПАНИИ / HR ATTRITION)
# =============================================================================
print("--> Шаг 1: Создание HR-датасета (уйдет ли сотрудник из компании)")

np.random.seed(42)
n_samples = 1000

# Признаки:
# 1. Удовлетворенность работой (от 0.0 до 1.0)
satisfaction = np.random.uniform(0.1, 1.0, size=n_samples)

# 2. Оценка руководителя (от 0.3 до 1.0)
evaluation = np.random.uniform(0.3, 1.0, size=n_samples)

# 3. Количество проектов (от 2 до 7)
projects = np.random.randint(2, 8, size=n_samples)

# 4. Среднее количество рабочих часов в месяц (120 - 300)
monthly_hours = np.random.uniform(120, 310, size=n_samples)

# 5. Стаж в компании в годах (1 - 10)
years_at_company = np.random.randint(1, 11, size=n_samples)

# 6. Было ли повышение за последние 5 лет (0 или 1)
promotion = (np.random.rand(n_samples) < 0.08).astype(int)

# Сложный паттерн ухода (1 = уволился, 0 = остался):
# Уходят ЛИБО те, кто перегорел (низкая удовлетворенность + переработки > 240 часов),
# ЛИБО талантливые сотрудники, которых не повышают (оценка > 0.8, стаж > 4 года, но promotion=0)
churn_prob = (
    0.6 * (satisfaction < 0.4) +
    0.3 * (monthly_hours > 250) +
    0.3 * ((evaluation > 0.8) & (years_at_company > 4) & (promotion == 0)) +
    np.random.normal(0, 0.15, n_samples)
)
left_company = (churn_prob > 0.5).astype(int)

df = pd.DataFrame({
    'satisfaction': np.round(satisfaction, 2),
    'evaluation': np.round(evaluation, 2),
    'projects': projects,
    'monthly_hours': np.round(monthly_hours, 1),
    'years_at_company': years_at_company,
    'promotion': promotion,
    'left_company': left_company
})

print("\nПервые 5 строк:")
print(df.head())
print(f"\nДоля уволившихся: {df['left_company'].mean() * 100:.1f}%")


# =============================================================================
# ШАГ 2: РАЗДЕЛЕНИЕ НА TRAIN И TEST
# =============================================================================
print("\n--> Шаг 2: Разделение данных")
    
X = df.drop(columns=['left_company'])
y = df['left_company']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} строк, Test: {X_test.shape[0]} строк")


# =============================================================================
# ШАГ 3: ОБУЧЕНИЕ СЛУЧАЙНОГО ЛЕСА
# =============================================================================
print("\n--> Шаг 3: Обучение RandomForestClassifier (150 деревьев)")

# Ключевые гиперпараметры:
# - n_estimators: количество деревьев в ансамбле (обычно от 100 до 500)
# - max_depth: глубина деревьев (None или ограничение 8-15)
# - n_jobs=-1: задействовать ВСЕ ядра процессора компьютера!
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("Ансамбль успешно обучен!")


# =============================================================================
# ШАГ 4: ОЦЕНКА КАЧЕСТВА
# =============================================================================
print("\n--> Шаг 4: Оценка на тестовых данных")

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Точность (Accuracy): {acc * 100:.2f}%")
print(f"ROC-AUC:            {roc_auc:.4f}")
print("\nПодробный отчет по классам:")
print(classification_report(y_test, y_pred, target_names=["Остался", "Уволился"]))


# =============================================================================
# ШАГ 5: ВАЖНОСТЬ ПРИЗНАКОВ (FEATURE IMPORTANCE)
# =============================================================================
print("\n--> Шаг 5: Анализ важности факторов для ухода")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
for col, imp in importances.items():
    print(f"  {col:20s}: {imp * 100:.2f}%")


# =============================================================================
# ШАГ 6: ИНФЕРЕНС (Проверка риска ухода конкретного специалиста)
# =============================================================================
print("\n--> Шаг 6: Проверка сотрудника на риск выгорания/ухода")

employee = pd.DataFrame([{
    'satisfaction': 0.28,
    'evaluation': 0.88,
    'projects': 5,
    'monthly_hours': 265.0,
    'years_at_company': 5,
    'promotion': 0
}])

churn_risk = rf.predict_proba(employee)[0][1]
decision = rf.predict(employee)[0]

print(f"Параметры: Удовлетворенность 0.28, Часов в месяц 265, Стаж 5 лет, Повышений нет")
print(f"Вероятность ухода: {churn_risk * 100:.1f}%")
print(f"Вердикт системы: {'ТРЕБУЕТСЯ ВНИМАНИЕ HR (Высокий риск ухода)' if decision == 1 else 'Всё в порядке'}")


# =============================================================================
# ШАГ 7: ВИЗУАЛИЗАЦИЯ (Важность признаков + Влияние числа деревьев)
# =============================================================================
print("\n--> Шаг 7: Построение графиков")

plt.figure(figsize=(14, 5))

# График 1: Важность признаков
plt.subplot(1, 2, 1)
y_positions = np.arange(len(importances))
plt.barh(y_positions, importances.values[::-1], color='teal', edgecolor='black')
plt.yticks(y_positions, importances.index[::-1])
plt.xlabel("Важность признака (Gini Importance)")
plt.title("Случайный лес: Что больше всего влияет на уход?")
plt.grid(True, linestyle=":", alpha=0.6)

# График 2: Как число деревьев влияет на качество модели
plt.subplot(1, 2, 2)
tree_counts = [1, 5, 10, 25, 50, 100, 150, 200]
accuracies = []

for n in tree_counts:
    temp_rf = RandomForestClassifier(n_estimators=n, max_depth=8, random_state=42, n_jobs=-1)
    temp_rf.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, temp_rf.predict(X_test)))

plt.plot(tree_counts, accuracies, marker='o', color='darkgreen', lw=2)
plt.title("Мудрость толпы: Рост качества от числа деревьев")
plt.xlabel("Количество деревьев в лесу (n_estimators)")
plt.ylabel("Точность на Test (Accuracy)")
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plot_path = PLOTS_DIR / "04_random_forest.png"
plt.savefig(plot_path, dpi=120)
plt.close()

print(f"График сохранен в файл: {plot_path}")
print("\n[Готово!] Запустите следующий скрипт: 05_gradient_boosting.py")
