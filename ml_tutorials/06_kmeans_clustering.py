"""
===============================================================================
TUTORIAL 06: K-MEANS CLUSTERING (Кластеризация К-средних)
===============================================================================

1. ЧТО ЭТО И ЗАЧЕМ НУЖНО?
------------------------
До сих пор мы рассматривали ОБУЧЕНИЕ С УЧИТЕЛЕМ (Supervised Learning), где у нас
всегда была известная правильная цель (y) — цена, увольнение, одобрение кредита.

K-Means — это представитель ОБУЧЕНИЯ БЕЗ УЧИТЕЛЯ (Unsupervised Learning).
Здесь НЕТ целевой переменной y! Модель сама находит скрытые группы (кластеры)
в данных на основе их схожести.
Типичные применения:
    - Сегментация клиентов магазина/сервиса для таргетированной рекламы.
    - Поиск аномалий (подозрительные транзакции, не похожие ни на один кластер).
    - Сжатие изображений и группировка похожих товаров.

2. КАК ЭТО РАБОТАЕТ ВНУТРИ?
---------------------------
Алгоритм группирует точки вокруг K центров (центроидов):
  1. Случайно выбирает K стартовых центроидов.
  2. Каждому клиенту назначается ближайший центроид (по евклидову расстоянию).
  3. Центроид пересчитывается: он перемещается в среднюю геометрическую точку всех
     своих клиентов.
  4. Шаги 2 и 3 повторяются, пока центроиды не перестанут двигаться.

3. КРИТИЧЕСКИ ВАЖНО:
--------------------
Поскольку K-Means считает геометрические расстояния между точками, МАСШТАБИРОВАНИЕ
(StandardScaler) — ОБЯЗАТЕЛЬНО! Если доход измеряется в миллионах, а возраст в годах,
алгоритм будет учитывать только доход, полностью игнорируя возраст.

4. КАК ВЫБРАТЬ ЧИСЛО КЛАСТЕРОВ K?
---------------------------------
- Метод локтя (Elbow Method): график падения инерции (суммы расстояний до центров).
  Ищем точку изгиба («локоть»), после которой ошибка падает медленно.
- Silhouette Score (Коэффициент силуэта): число от -1 до +1. Чем выше, тем плотнее
  и изолированнее кластеры.
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ (СЕГМЕНТАЦИЯ КЛИЕНТОВ ИНТЕРНЕТ-МАГАЗИНА)
# =============================================================================
print("--> Шаг 1: Создание клиентской базы интернет-магазина (без разметки)")

np.random.seed(42)

# Создадим 4 естественные группы клиентов:
# 1. Экономные: низкий доход, низкие траты
c1 = np.random.normal(loc=[30, 20], scale=[8, 8], size=(120, 2))
# 2. Шопоголики-студенты: невысокий доход, но максимальные траты
c2 = np.random.normal(loc=[35, 80], scale=[8, 8], size=(100, 2))
# 3. Рациональные богачи: высокий доход, умеренные траты
c3 = np.random.normal(loc=[95, 30], scale=[10, 8], size=(110, 2))
# 4. VIP / Премиум: высокий доход, высокие траты
c4 = np.random.normal(loc=[105, 85], scale=[10, 7], size=(120, 2))

all_data = np.vstack([c1, c2, c3, c4])

# Ограничим значения логичными рамками
annual_income = np.clip(all_data[:, 0], 15, 140)
spending_score = np.clip(all_data[:, 1], 1, 99)

df = pd.DataFrame({
    'annual_income_k_usd': np.round(annual_income, 1),
    'spending_score_1_to_100': np.round(spending_score, 1)
})

print("\nПервые 5 клиентов (только признаки, никаких классов!):")
print(df.head())


# =============================================================================
# ШАГ 2: ОБЯЗАТЕЛЬНОЕ МАСШТАБИРОВАНИЕ ДАННЫХ
# =============================================================================
print("\n--> Шаг 2: Стандартизация признаков (StandardScaler)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print("Данные центрированы со средним 0 и стандартным отклонением 1.")


# =============================================================================
# ШАГ 3: ПОИСК ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ (МЕТОД ЛОКТЯ / ELBOW METHOD)
# =============================================================================
print("\n--> Шаг 3: Поиск оптимального числа кластеров K (от 2 до 8)")

k_range = range(2, 9)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)  # Сумма квадратов расстояний до центроидов
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(f"  K = {k:d} | Инерция: {km.inertia_:7.1f} | Silhouette: {silhouettes[-1]:.3f}")

# Выбираем K=4, так как оно дает четкий пик Silhouette и изгиб локтя
optimal_k = 4
print(f"\nОптимальное количество сегментов: K = {optimal_k}")


# =============================================================================
# ШАГ 4: ФИНАЛЬНАЯ КЛАСТЕРИЗАЦИЯ
# =============================================================================
print(f"\n--> Шаг 4: Обучение K-Means с K={optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Центры кластеров в исходных шкалах (доллары и баллы)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

print("\nХарактеристики найденных сегментов аудитории:")
for i, center in enumerate(centroids_original):
    count = (df['cluster'] == i).sum()
    print(f"  Кластер {i} (размер: {count} чел.): "
          f"Средний доход = ${center[0]:.1f}k, Балл трат = {center[1]:.1f}")


# =============================================================================
# ШАГ 5: ИНФЕРЕНС (К какому сегменту отнести нового посетителя?)
# =============================================================================
print("\n--> Шаг 5: Сегментация нового посетителя сайта")

new_client = pd.DataFrame([{
    'annual_income_k_usd': 110.0,
    'spending_score_1_to_100': 88.0
}])

new_client_scaled = scaler.transform(new_client)
assigned_cluster = kmeans.predict(new_client_scaled)[0]

print(f"Новый клиент: Доход $110k, Балл трат 88")
print(f"Отнесен к сегменту: Кластер {assigned_cluster} (Премиум-шопоголик -> персональный VIP-менеджер)")


# =============================================================================
# ШАГ 6: ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ И МЕТОДА ЛОКТЯ
# =============================================================================
print("\n--> Шаг 6: Построение графиков")

plt.figure(figsize=(14, 5))

# График 1: Метод локтя и Silhouette Score
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-', lw=2, label='Инерция (падение ошибки)')
plt.axvline(x=4, color='red', linestyle='--', label='Локоть (K=4)')
plt.title("Выбор числа кластеров: Метод локтя (Elbow Method)")
plt.xlabel("Количество кластеров (K)")
plt.ylabel("Инерция (Сумма расстояний)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

# График 2: Карта сегментов клиентов
plt.subplot(1, 2, 2)
colors = ['purple', 'forestgreen', 'darkorange', 'dodgerblue']

for i in range(optimal_k):
    cluster_points = df[df['cluster'] == i]
    plt.scatter(
        cluster_points['annual_income_k_usd'],
        cluster_points['spending_score_1_to_100'],
        c=colors[i],
        label=f'Сегмент {i}',
        alpha=0.6,
        edgecolors='none'
    )

# Рисуем центроиды красными крестиками
plt.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Центроиды групп',
    edgecolors='black',
    linewidths=1.5
)

plt.title("K-Means: 4 выявленных сегмента покупателей")
plt.xlabel("Годовой доход ($ в тысячах)")
plt.ylabel("Балл активности трат (1-100)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left")

plt.tight_layout()
plot_path = PLOTS_DIR / "06_kmeans_clustering.png"
plt.savefig(plot_path, dpi=120)
plt.close()

print(f"График сохранен в файл: {plot_path}")
print("\n[Готово!] Запустите следующий скрипт: 07_production_pipeline.py")
