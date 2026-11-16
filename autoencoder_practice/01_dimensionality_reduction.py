"""
================================================================================
ПРИМЕР 1: АВТОЭНКОДЕР ДЛЯ СНИЖЕНИЯ РАЗМЕРНОСТИ (DIMENSIONALITY REDUCTION)
================================================================================

В этой задаче мы изучаем фундаментальную роль автоэнкодера:
Сжатие высокоразмерного пространства признаков в компактное латентное представление (Bottleneck).

ЧТО ТАКОЕ АВТОЭНКОДЕР (AUTOENCODER):
Это нейросеть с симметричной архитектурой «песочных часов» (Hourglass):
  Вход X (64D) ──► [Encoder] ──► Скрытый код Z (2D) ──► [Decoder] ──► Выход X̂ (64D)

  1. Энкодер (Encoder): сжимает данные через последовательное уменьшение слоев
     (64 -> 32 -> 16 -> 2).
  2. Бутылочное горлышко (Bottleneck / Latent Space Z): узкий слой минимальной
     размерности (например, 2 числа вместо 64). В нем сохраняются только самые
     важные закономерности и взаимосвязи.
  3. Декодер (Decoder): разворачивает сжатый код Z обратно в исходный размер
     (2 -> 16 -> 32 -> 64).
  4. Целевая функция (Reconstruction Loss): сеть учится так, чтобы выход X̂ был
     максимально похож на вход X: L = MSE(X, X̂).

В ЧЕМ ПРЕИМУЩЕСТВО ПЕРЕД PCA (Principal Component Analysis):
- PCA строит только ЛИНЕЙНЫЕ комбинации признаков (гиперплоскости).
- Автоэнкодер за счет функций активации (GELU, ReLU) находит СЛОЖНЫЕ НЕЛИНЕЙНЫЕ
  многообразия (кривые, спирали, замкнутые поверхности), которые линейный PCA
  сминает и теряет.
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


# ==============================================================================
# 1. ГЕНЕРАЦИЯ ВЫСОКОРАЗМЕРНЫХ СЕНСОРНЫХ ДАННЫХ СО СЛОЖНЫМИ СВЯЗЯМИ
# ==============================================================================
def generate_multivariate_sensor_data(n_samples=2500, n_features=64, random_seed=42):
    """
    Генерирует 64-мерный вектор сенсоров промышленной установки.
    В основе лежат 3 скрытых режима работы (кластера) на нелинейной спирали,
    которые затем проецируются в 64 зашумленных канала измерений.
    """
    np.random.seed(random_seed)
    n_per_cluster = n_samples // 3

    # Нелинейное скрытое 2D-многообразие (3 переплетенных рукава)
    latent_points = []
    labels = []
    for cluster_id in range(3):
        phi = np.linspace(0, 1.8 * np.pi, n_per_cluster) + (cluster_id * 2 * np.pi / 3)
        radius = np.linspace(1.5, 6.0, n_per_cluster) + np.random.normal(0, 0.2, n_per_cluster)
        x_latent = radius * np.cos(phi)
        y_latent = radius * np.sin(phi)
        coords = np.column_stack([x_latent, y_latent])
        latent_points.append(coords)
        labels.extend([cluster_id] * n_per_cluster)

    latent_2d = np.vstack(latent_points)
    labels = np.array(labels)

    # Нелинейное проецирование из 2D в 64D с добавлением датчикового шума
    random_projection = np.random.randn(2, n_features)
    non_linear_manifold = np.sin(latent_2d @ random_projection) * 3.0 + (latent_2d[:, :1] ** 2) * 0.05
    sensor_noise = np.random.normal(0, 0.25, size=non_linear_manifold.shape)
    
    high_dim_data = non_linear_manifold + sensor_noise
    return high_dim_data.astype(np.float32), latent_2d, labels


# ==============================================================================
# 2. АРХИТЕКТУРА АВТОЭНКОДЕРА (PYTORCH)
# ==============================================================================
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=2):
        super().__init__()
        
        # 1. Энкодер: сжимает 64 -> 32 -> 16 -> 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, latent_dim)  # Бутылочное горлышко (Bottleneck)
        )
        
        # 2. Декодер: восстанавливает 2 -> 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        """Возвращает сжатое латентное представление (код Z)"""
        return self.encoder(x)

    def decode(self, z):
        """Восстанавливает исходные данные из кода Z"""
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ==============================================================================
# 3. ОСНОВНОЙ ПАЙПЛАЙН: ОБУЧЕНИЕ, СРАВНЕНИЕ С PCA И ВИЗУАЛИЗАЦИЯ
# ==============================================================================
def run_dimensionality_reduction_pipeline():
    print("=" * 70)
    print("📉 1. АВТОЭНКОДЕР ДЛЯ СНИЖЕНИЯ РАЗМЕРНОСТИ (64D -> 2D)")
    print("=" * 70)

    # 1. Данные
    X_raw, true_latent, labels = generate_multivariate_sensor_data(n_samples=3000, n_features=64, random_seed=42)
    X_train, X_test, labels_train, labels_test = train_test_split(
        X_raw, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"✓ Сгенерированы 64-мерные сенсорные профили: Train={len(X_train)}, Test={len(X_test)}")

    # 2. Классический метод: Линейный PCA (2 компоненты)
    print("\n[Базовый метод] Обучаем классический PCA(n_components=2)...")
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_test_pca_latent = pca.transform(X_test)
    X_test_pca_recon = pca.inverse_transform(X_test_pca_latent)
    pca_mse = mean_squared_error(X_test, X_test_pca_recon)
    print(f"✓ PCA(2D) ошибка реконструкции MSE: {pca_mse:.4f}")

    # 3. Нелинейный метод: Глубокий автоэнкодер (PyTorch)
    print("\n[Нейросеть] Обучаем Deep Autoencoder (64 -> 32 -> 16 -> 2 -> 16 -> 32 -> 64)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Вычисления на устройстве: {device}")

    model = DeepAutoencoder(input_dim=64, latent_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-4)

    # Превращаем в DataLoader
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=64, shuffle=True)

    epochs = 40
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, _ = model(batch_x)
            loss = criterion(recon_x, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Эпоха {epoch+1:2d}/{epochs:2d} | Потери (MSE Loss): {epoch_loss/len(X_train):.5f}")

    # 4. Оценка автоэнкодера на тестовой выборке
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        recon_test, latent_test = model(test_tensor)
        ae_mse = criterion(recon_test, test_tensor).item()
        ae_latent = latent_test.cpu().numpy()
        ae_recon = recon_test.cpu().numpy()

    print("\n" + "=" * 60)
    print(f"{'Метод сжатия (64D -> 2D)':<30} | {'Тестовая ошибка (MSE)':<20}")
    print("-" * 60)
    print(f"{'Линейный PCA (2 компоненты)':<30} | {pca_mse:12.4f}")
    print(f"{'Нелинейный Автоэнкодер (2D)':<30} | {ae_mse:12.4f}")
    print("=" * 60)
    reduction = ((pca_mse - ae_mse) / pca_mse) * 100
    print(f"🏆 Превосходство автоэнкодера: ошибка восстановления снижена на {reduction:.1f}%!")

    # 5. Визуализация латентных пространств и качества восстановления
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    cluster_names = ["Режим А (Высокая нагрузка)", "Режим Б (Номинал)", "Режим В (Холостой ход)"]

    # Панель 1: Латентное пространство PCA
    for c_id in range(3):
        mask = (labels_test == c_id)
        axes[0, 0].scatter(
            X_test_pca_latent[mask, 0], X_test_pca_latent[mask, 1],
            c=colors[c_id], label=cluster_names[c_id], alpha=0.6, s=35
        )
    axes[0, 0].set_title(f"1. Линейная проекция PCA (2 компоненты)\nMSE реконструкции = {pca_mse:.4f}", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("PC 1")
    axes[0, 0].set_ylabel("PC 2")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper right", fontsize=9)

    # Панель 2: Латентное пространство Автоэнкодера (Bottleneck)
    for c_id in range(3):
        mask = (labels_test == c_id)
        axes[0, 1].scatter(
            ae_latent[mask, 0], ae_latent[mask, 1],
            c=colors[c_id], label=cluster_names[c_id], alpha=0.6, s=35
        )
    axes[0, 1].set_title(f"2. Нелинейное латентное пространство Автоэнкодера (Bottleneck Z)\nMSE реконструкции = {ae_mse:.4f}", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Latent Dim Z1")
    axes[0, 1].set_ylabel("Latent Dim Z2")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="upper right", fontsize=9)

    # Панель 3: Сравнение восстановления 64 признаков для одного тестового образца
    sample_idx = 10
    feature_indices = np.arange(64)
    axes[1, 0].plot(feature_indices, X_test[sample_idx], label="Истинный сенсорный вектор (Ground Truth)", color="black", lw=2.2)
    axes[1, 0].plot(feature_indices, X_test_pca_recon[sample_idx], label=f"Восстановление PCA [MSE={pca_mse:.3f}]", color="#e67e22", ls="--", lw=1.8)
    axes[1, 0].plot(feature_indices, ae_recon[sample_idx], label=f"Восстановление Автоэнкодера [MSE={ae_mse:.3f}]", color="#27ae60", lw=2.0)
    axes[1, 0].set_title(f"3. Профиль 64 сенсоров (Сравнение восстановления образца #{sample_idx})", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Номер датчика (Канал 0..63)")
    axes[1, 0].set_ylabel("Значение сигнала")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="upper right", fontsize=9)

    # Панель 4: Распределение ошибок восстановления по всем тестам
    ae_sample_errors = np.mean((X_test - ae_recon) ** 2, axis=1)
    pca_sample_errors = np.mean((X_test - X_test_pca_recon) ** 2, axis=1)
    axes[1, 1].hist(pca_sample_errors, bins=35, alpha=0.5, color="#e67e22", label=f"Ошибки PCA (Среднее: {pca_mse:.4f})")
    axes[1, 1].hist(ae_sample_errors, bins=35, alpha=0.6, color="#27ae60", label=f"Ошибки Автоэнкодера (Среднее: {ae_mse:.4f})")
    axes[1, 1].set_title("4. Гистограмма ошибок восстановления на тесте (Sample-level MSE)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("MSE ошибки")
    axes[1, 1].set_ylabel("Количество образцов")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join("plots", "01_dimensionality_reduction.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n📊 График сохранен в файл: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_dimensionality_reduction_pipeline()
