"""
===============================================================================
TUTORIAL 03: DECISION TREES (Деревья решений)
===============================================================================

1. ЧТО ЭТО И ЗАЧЕМ НУЖНО?
------------------------
Дерево решений — один из самых интуитивно понятных алгоритмов.
Оно строит систему последовательных правил «ЕСЛИ-ТО»:
    ЕСЛИ возраст > 30:
        ЕСЛИ доход > 100k: Одобрить
        ИНАЧЕ: Отказать
    ИНАЧЕ: Отказать

Деревья умеют решать как классификацию (DecisionTreeClassifier),
так и регрессию (DecisionTreeRegressor).

2. КАК ЭТО РАБОТАЕТ ВНУТРИ?
---------------------------
В каждом узле (Node) алгоритм перебирает ВСЕ признаки и ВСЕ пороги разбиения,
выбирая то условие, которое лучше всего разделяет классы (уменьшает «хаос»).
Мера хаоса обычно измеряется через критерий Джини (Gini Impurity) или Энтропию.
    - Джини = 0: узел «чистый» (все примеры в нем одного класса).
    - Джини = 0.5: максимальный хаос (50% одного класса, 50% другого).

3. ГЛАВНАЯ ОПАСНОСТЬ: ПЕРЕОБУЧЕНИЕ (OVERFITTING)!
-------------------------------------------------
Если не ограничить глубину дерева (параметр max_depth), дерево вырастет настолько,
что создаст отдельный листик почти под каждую строчку обучающей выборки.
Результат:
    - Точность на Train = 100% (модель просто вызубрила примеры наизусть)
    - Точность на Test = Ужасная (на новых данных модель беспомощна)
Лекарство: Ограничение глубины дерева (max_depth=3..5).

4. ПЛЮСЫ И МИНУСЫ:
------------------
(+) Не требует нормализации/масштабирования данных (StandardScaler не нужен!).
(+) Легко визуализировать и объяснить неспециалисту (White-Box модель).
(+) Легко улавливает нелинейные зависимости.
(-) Одиночное дерево нестабильно (высокая дисперсия, склонно к переобучению).
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ (ПОКУПКА ПРЕМИАЛЬНОГО АБОНЕМЕНТА В СПОРТЗАЛ)
# =============================================================================
print("--> Шаг 1: Создание нелинейного датасета о клиентах фитнес-клуба")

np.random.seed(42)
n_samples = 600

# Признаки:
# 1. Возраст (18 - 65)
age = np.random.randint(18, 65, size=n_samples)
# 2. Количество визитов в месяц (0 - 25)
monthly_visits = np.random.randint(0, 25, size=n_samples)
# 3. Средний чек в баре фитнес-клуба в рублях (0 - 5000)
bar_spending = np.random.uniform(0, 5000, size=n_samples)

# Нелинейная логика покупки премиум-статуса (1 = купил, 0 = нет):
# Покупают: ЛИБО частые посетители (визиты >= 12), ЛИБО люди старше 35 с высокими тратами в баре
bought_vip = (
    (monthly_visits >= 12) |
    ((age >= 35) & (bar_spending >= 2500))
).astype(int)

# Добавим немного случайного шума (5% людей меняют решение непредсказуемо)
flip_mask = np.random.rand(n_samples) < 0.05
bought_vip[flip_mask] = 1 - bought_vip[flip_mask]

df = pd.DataFrame({
    'age': age,
    'monthly_visits': monthly_visits,
    'bar_spending': np.round(bar_spending, 0),
    'bought_vip': bought_vip
})

print("\nПервые 5 клиентов:")
print(df.head())


# =============================================================================
# ШАГ 2: РАЗДЕЛЕНИЕ НА TRAIN И TEST
# =============================================================================
print("\n--> Шаг 2: Разделение данных")

X = df[['age', 'monthly_visits', 'bar_spending']]
y = df['bought_vip']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
# Деревья не требуют масштабирования данных! Оставляем числа в их исходных единицах.


# =============================================================================
# ШАГ 3: ДЕМОНСТРАЦИЯ ПЕРЕОБУЧЕНИЯ (ГЛУБИНА ДЕРЕВА)
# =============================================================================
print("\n--> Шаг 3: Сравнение переобученного дерева и сбалансированного")

# Вариант А: Без ограничений (max_depth=None) — ПЕРЕОБУЧЕНИЕ!
overfitted_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
overfitted_tree.fit(X_train, y_train)

train_acc_over = accuracy_score(y_train, overfitted_tree.predict(X_train))
test_acc_over = accuracy_score(y_test, overfitted_tree.predict(X_test))

print(f"[Без ограничений] Глубина: {overfitted_tree.get_depth()}")
print(f"  Точность на Train: {train_acc_over * 100:.1f}% (вызубрил)")
print(f"  Точность на Test:  {test_acc_over * 100:.1f}% (ошибается на новых!)")

# Вариант Б: Ограниченная глубина (max_depth=3) — ПРАВИЛЬНЫЙ БАЛАНС
optimal_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
optimal_tree.fit(X_train, y_train)

train_acc_opt = accuracy_score(y_train, optimal_tree.predict(X_train))
test_acc_opt = accuracy_score(y_test, optimal_tree.predict(X_test))

print(f"\n[Оптимальное дерево] Глубина: 3")
print(f"  Точность на Train: {train_acc_opt * 100:.1f}%")
print(f"  Точность на Test:  {test_acc_opt * 100:.1f}% (отличная обобщающая способность!)")


# =============================================================================
# ШАГ 4: ВАЖНОСТЬ ПРИЗНАКОВ (FEATURE IMPORTANCES)
# =============================================================================
print("\n--> Шаг 4: Какие признаки оказались самыми важными для дерева?")
# Сумма всех важностей всегда равна 1.0 (100%)
importances = optimal_tree.feature_importances_
for col, imp in zip(X.columns, importances):
    print(f"  {col:15s}: {imp * 100:.1f}%")


# =============================================================================
# ШАГ 5: ИНФЕРЕНС (Проверка нового клиента)
# =============================================================================
print("\n--> Шаг 5: Предсказание для нового посетителя")
new_guest = pd.DataFrame([{
    'age': 40,
    'monthly_visits': 4,
    'bar_spending': 3200
}])

pred = optimal_tree.predict(new_guest)[0]
prob = optimal_tree.predict_proba(new_guest)[0][1]
print(f"Клиент: 40 лет, 4 визита в месяц, тратит 3200 руб. в баре")
print(f"Купит VIP-карту? {'ДА' if pred == 1 else 'НЕТ'} (вероятность: {prob * 100:.1f}%)")


# =============================================================================
# ШАГ 6: ВИЗУАЛИЗАЦИЯ ДЕРЕВА И КРИВОЙ ПЕРЕОБУЧЕНИЯ
# =============================================================================
print("\n--> Шаг 6: Построение графиков")

plt.figure(figsize=(15, 6))

# График 1: Визуализация дерева решений (реальная блок-схема правил!)
plt.subplot(1, 2, 1)
plot_tree(
    optimal_tree,
    feature_names=X.columns,
    class_names=['Обычный', 'VIP'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Дерево решений (max_depth=3):\nНаглядная логика каждого правила")

# График 2: Зависимость качества от глубины (демонстрация Overfitting)
plt.subplot(1, 2, 2)
depths = range(1, 15)
train_scores = []
test_scores = []

for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, t.predict(X_train)))
    test_scores.append(accuracy_score(y_test, t.predict(X_test)))

plt.plot(depths, train_scores, 'o-', color='blue', label='Train Accuracy (Обучение)')
plt.plot(depths, test_scores, 's-', color='red', label='Test Accuracy (Тест)')
plt.axvline(x=3, color='green', linestyle='--', label='Оптимальная глубина (3)')
plt.title("Проблема переобучения (Overfitting):\nСлишком глубокое дерево зубрит Train, но ухудшает Test")
plt.xlabel("Глубина дерева (max_depth)")
plt.ylabel("Точность (Accuracy)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()
plot_path = PLOTS_DIR / "03_decision_tree.png"
plt.savefig(plot_path, dpi=120)
plt.close()

print(f"График сохранен в файл: {plot_path}")
print("\n[Готово!] Запустите следующий скрипт: 04_random_forest.py")
