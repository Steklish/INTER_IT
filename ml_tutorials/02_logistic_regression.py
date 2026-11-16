"""
===============================================================================
TUTORIAL 02: LOGISTIC REGRESSION (Логистическая регрессия)
===============================================================================

1. ЧТО ЭТО И ЗАЧЕМ НУЖНО?
------------------------
Несмотря на слово «регрессия» в названии, эта модель решает задачу
КЛАССИФИКАЦИИ (Classification): предсказание категорий или классов.
Самый частый случай — БИНАРНАЯ классификация (0 или 1, Да или Нет):
    - Одобрить кредит или отказать?
    - Уйдет ли клиент из сервиса (Churn)?
    - Спам письмо или нет?
    - Есть ли заболевание у пациента?

2. КАК ЭТО РАБОТАЕТ ВНУТРИ?
---------------------------
Почему нельзя использовать обычную линейную регрессию?
Линейная регрессия выдает числа от минус до плюс бесконечности (например, -15 или 250).
А вероятность ДОЛЖНА быть строго от 0.0 до 1.0 (0% до 100%)!

Логистическая регрессия берет линейную комбинацию z = (w * x + b)
и пропускает её через сигмоиду (Sigmoid function):
    p = 1 / (1 + e^(-z))

Сигмоида сжимает любое число в диапазон [0, 1] — получается вероятность!
По умолчанию, если p >= 0.5, модель ставит класс 1, иначе класс 0.

3. ПЛЮСЫ И МИНУСЫ:
------------------
(+) Выдает не просто класс, а калиброванную ВЕРОЯТНОСТЬ (уверенность модели).
(+) Быстрая, стабильная, отличный базовый ориентир (baseline).
(-) Строит только линейную границу разделения классов.
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ (ОДОБРЕНИЕ БАНКОВСКОГО КРЕДИТА)
# =============================================================================
print("--> Шаг 1: Создание датасета банковских заявок на кредит")

np.random.seed(42)
n_samples = 500

# Признаки клиента:
# 1. Возраст (от 18 до 70 лет)
age = np.random.randint(18, 70, size=n_samples)

# 2. Месячный доход в тысячах рублей (от 20 до 250 тыс.)
income = np.random.uniform(20, 250, size=n_samples)

# 3. Кредитный рейтинг (от 300 до 850 баллов)
credit_score = np.random.randint(300, 850, size=n_samples)

# Логика одобрения кредита (вероятность зависит от дохода и рейтинга):
# Высокий доход и высокий рейтинг сильно увеличивают вероятность одобрения
z = -6.0 + (income * 0.025) + (credit_score * 0.008) + (age * 0.01) + np.random.normal(0, 0.8, n_samples)
prob = 1 / (1 + np.exp(-z))
loan_approved = (prob >= 0.5).astype(int)  # 1 = одобрен, 0 = отказ

df = pd.DataFrame({
    'age': age,
    'income_k_rub': np.round(income, 1),
    'credit_score': credit_score,
    'loan_approved': loan_approved
})

print("\nПервые 5 клиентов (df.head()):")
print(df.head())

print(f"\nБаланс классов в датасете:")
print(df['loan_approved'].value_counts(normalize=True).rename({0: "0 (Отказ)", 1: "1 (Одобрен)"}))


# =============================================================================
# ШАГ 2: РАЗДЕЛЕНИЕ НА TRAIN / TEST И СТАНДАРТИЗАЦИЯ (SCALING)
# =============================================================================
print("\n--> Шаг 2: Разделение данных и масштабирование признаков")

X = df[['age', 'income_k_rub', 'credit_score']]
y = df['loan_approved']

# stratify=y гарантирует, что процент одобрений будет одинаковым и в Train, и в Test!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ВАЖНО ДЛЯ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ:
# Возраст (18..70) и Рейтинг (300..850) имеют разные масштабы.
# StandardScaler приводит каждый признак к среднему 0 и дисперсии 1.
# Заметьте: fit_transform делаем ТОЛЬКО на train, а на test — только transform!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================================================================
# ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ
# =============================================================================
print("\n--> Шаг 3: Обучение LogisticRegression")

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

print("Модель обучена!")
print("\nВажность признаков (веса модели):")
for col, weight in zip(X.columns, model.coef_[0]):
    direction = "увеличивает шанс" if weight > 0 else "уменьшает шанс"
    print(f"  {col:15s}: {weight:+.3f} ({direction})")


# =============================================================================
# ШАГ 4: ПРЕДСКАЗАНИЕ И КЛЮЧЕВЫЕ МЕТРИКИ КЛАССИФИКАЦИИ
# =============================================================================
print("\n--> Шаг 4: Оценка качества классификации")

# 1. Предсказание меток классов (0 или 1)
y_pred = model.predict(X_test_scaled)

# 2. Предсказание вероятностей (получаем массив [P(y=0), P(y=1)])
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Метрики:
# - Accuracy: общая доля правильных ответов (может обманывать при дисбалансе классов!)
# - Precision (Точность): из тех, кому одобрили, скольким действительно стоило одобрить?
# - Recall (Полнота): какой процент надежных клиентов мы вообще нашли?
# - F1-Score: гармоническое среднее между Precision и Recall
# - ROC-AUC: способность модели ранжировать клиентов от самых ненадежных к надежным (0.5 = монетка, 1.0 = идеал)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy  (Общая точность): {acc:.3f}")
print(f"Precision (Точность '1'):   {prec:.3f}")
print(f"Recall    (Полнота '1'):    {rec:.3f}")
print(f"F1-Score  (Баланс P и R):   {f1:.3f}")
print(f"ROC-AUC   (Ранжирование):   {roc_auc:.3f}")

print("\nПодробный отчет (Classification Report):")
print(classification_report(y_test, y_pred, target_names=["Отказ (0)", "Одобрен (1)"]))


# =============================================================================
# ШАГ 5: ИНФЕРЕНС ДЛЯ НОВОГО КЛИЕНТА
# =============================================================================
print("\n--> Шаг 5: Проверка нового клиента")

client = pd.DataFrame([{
    'age': 32,
    'income_k_rub': 110.0,
    'credit_score': 720
}])

# Масштабируем так же, как обучающие данные!
client_scaled = scaler.transform(client)
decision = model.predict(client_scaled)[0]
approval_probability = model.predict_proba(client_scaled)[0][1]

print(f"Клиент: Возраст 32, Доход 110 тыс. руб, Рейтинг 720")
print(f"Вероятность одобрения: {approval_probability * 100:.1f}%")
print(f"Решение банка: {'ОДОБРЕНО' if decision == 1 else 'ОТКАЗ'}")


# =============================================================================
# ШАГ 6: ВИЗУАЛИЗАЦИЯ (Матрица ошибок + ROC-кривая)
# =============================================================================
print("\n--> Шаг 6: Построение графиков")

plt.figure(figsize=(12, 5))

# График 1: Матрица ошибок (Confusion Matrix)
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Матрица ошибок (Confusion Matrix)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Отказ (0)", "Одобрен (1)"])
plt.yticks(tick_marks, ["Отказ (0)", "Одобрен (1)"])

# Текстовые подписи в ячейках
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

plt.ylabel("Истинный класс")
plt.xlabel("Предсказанный класс")

# График 2: ROC-кривая
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный выбор (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Ложная тревога)')
plt.ylabel('True Positive Rate (Чувствительность)')
plt.title('ROC-кривая (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plot_path = PLOTS_DIR / "02_logistic_regression.png"
plt.savefig(plot_path, dpi=120)
plt.close()

print(f"График сохранен в файл: {plot_path}")
print("\n[Готово!] Запустите следующий скрипт: 03_decision_tree.py")
