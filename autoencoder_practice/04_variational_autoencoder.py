"""
================================================================================
ПРИМЕР 4: ВАРИАЦИОННЫЙ АВТОЭНКОДЕР (VARIATIONAL AUTOENCODER, VAE)
================================================================================

В этой задаче мы изучаем переход от простого сжатия к ГЕНЕРАТИВНОМУ ИИ:
Вариационный автоэнкодер (VAE) — фундамент современной диффузии и генеративных моделей.

ПОЧЕМУ ОБЫЧНЫЙ АВТОЭНКОДЕР НЕ УМЕЕТ ГЕНЕРИРОВАТЬ НОВЫЕ ОБЪЕКТЫ:
В обычном автоэнкодере латентное пространство Z дискретно. Точки разных классов
кучкуются в изолированные "островки", а между ними лежат огромные "пустоты".
Если взять случайную точку из пустоты и отправить в декодер, получится бессмысленный шум.

КАК VAE ДЕЛАЕТ ЛАТЕНТНОЕ ПРОСТРАНСТВО НЕПРЕРЫВНЫМ:
1. Вместо одной точки Z энкодер предсказывает параметры нормального распределения:
   - Вектор средних: mu
   - Вектор логарифмов дисперсии: log(sigma^2)

2. Трюк перепараметризации (Reparameterization Trick):
   Чтобы через случайный процесс могли протекать градиенты (Backpropagation),
   мы семплируем шум epsilon ~ N(0, I) отдельно:
       z = mu + sigma * epsilon

3. Двойная функция потерь (VAE Loss):
   Loss = Reconstruction_Loss (MSE) + beta * KL_Divergence
   - Reconstruction Loss заставляет декодер качественно восстанавливать сигнал.
   - Дивергенция Кульбака-Лейблера (KL) притягивает распределения к стандартному N(0, 1).
   Она устраняет дыры и "пустоты", делая латентное пространство непрерывным и гладким!

ДВА КЛЮЧЕВЫХ ПРИМЕНЕНИЯ VAE:
1. Генерация абсолютно новых синтетических образцов: Декодер(z ~ N(0, I)).
2. Плавная интерполяция (Latent Walk / Морфинг): непрерывный переход z_A -> z_B.
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


# ==============================================================================
# 1. ГЕНЕРАЦИЯ МУЛЬТИМОДАЛЬНЫХ ФИЗИЧЕСКИХ ПРОФИЛЕЙ (3 КЛАССА ФОРМ ВОЛНЫ)
# ==============================================================================
def generate_multiclass_profiles(n_samples=3000, profile_len=64, random_seed=42):
    """
    Генерирует 3 различных типа сигнатур технологического процесса (64 точки):
      - Класс 0: Двойной синусоидальный резонанс (высокие частоты)
      - Класс 1: Гауссовский импульсный колокол (нагрев)
      - Класс 2: Ступенчатый релаксационный спад (переходный процесс)
    """
    np.random.seed(random_seed)
    t = np.linspace(0, 1, profile_len)
    
    profiles = []
    labels = []

    for _ in range(n_samples // 3):
        # Класс 0: Резонанс
        freq = np.random.uniform(2.5, 3.5)
        amp = np.random.uniform(1.2, 1.8)
        sig0 = amp * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.08, profile_len)
        profiles.append(sig0)
        labels.append(0)

        # Класс 1: Колокол (Гауссиана)
        center = np.random.uniform(0.4, 0.6)
        width = np.random.uniform(0.08, 0.15)
        amp1 = np.random.uniform(2.0, 3.0)
        sig1 = amp1 * np.exp(-((t - center) ** 2) / (2 * width ** 2)) + np.random.normal(0, 0.08, profile_len)
        profiles.append(sig1)
        labels.append(1)

        # Класс 2: Ступенчатый спад (экспонента)
        decay = np.random.uniform(3.0, 6.0)
        amp2 = np.random.uniform(2.0, 2.8)
        sig2 = amp2 * np.exp(-decay * t) + np.random.normal(0, 0.08, profile_len)
        profiles.append(sig2)
        labels.append(2)

    return np.array(profiles, dtype=np.float32), np.array(labels, dtype=int)


# ==============================================================================
# 2. АРХИТЕКТУРА ВАРИАЦИОННОГО АВТОЭНКОДЕРА (VAE)
# ==============================================================================
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=2):
        super().__init__()
        
        # 1. Энкодер: сжимает 64 -> 32
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Две раздельные головы: вектор средних (mu) и вектор логарифма дисперсии (logvar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 2. Декодер: восстанавливает 2 -> 32 -> 64
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Трюк перепараметризации (Reparameterization Trick):
        z = mu + std * epsilon, где epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


# ==============================================================================
# 3. ФУНКЦИЯ ПОТЕРЬ VAE (RECONSTRUCTION + KL DIVERGENCE)
# ==============================================================================
def vae_loss_function(recon_x, x, mu, logvar, beta=0.002):
    """
    1. Reconstruction Loss (MSE): насколько точно декодер восстановил исходный сигнал.
    2. KL Divergence: штраф за отклонение латентного распределения от стандартного N(0, I).
       Формула KL для гауссианы: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ==============================================================================
# 4. ОСНОВНОЙ ПАЙПЛАЙН: ОБУЧЕНИЕ, СЕМПЛИРОВАНИЕ И ИНТЕРПОЛЯЦИЯ
# ==============================================================================
def run_vae_pipeline():
    print("=" * 70)
    print("🎨 4. ВАРИАЦИОННЫЙ АВТОЭНКОДЕР (VAE): ГЕНЕРАЦИЯ И МОРФИНГ")
    print("=" * 70)

    # 1. Данные
    X_raw, labels = generate_multiclass_profiles(n_samples=3000, profile_len=64, random_seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"✓ Датасет: 3000 профилей 3 классов (Резонанс, Колокол, Спад)")
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

    # 2. Обучение VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Вычисления на устройстве: {device}")

    model = VariationalAutoencoder(input_dim=64, hidden_dim=32, latent_dim=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-5)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=64, shuffle=True)

    epochs = 40
    print("\nОбучаем VAE (с оптимизацией ELBO: MSE + KL Divergence)...")
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar, _ = model(batch_x)
            loss, r_loss, kl_loss = vae_loss_function(recon_x, batch_x, mu, logvar, beta=0.005)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item() * len(batch_x)
            total_recon_loss += r_loss.item() * len(batch_x)
            total_kl_loss += kl_loss.item() * len(batch_x)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Эпоха {epoch+1:2d}/{epochs:2d} | Total Loss: {total_epoch_loss/len(X_train):.4f} "
                  f"(Recon MSE: {total_recon_loss/len(X_train):.4f}, KL: {total_kl_loss/len(X_train):.4f})")

    # 3. Извлечение латентного пространства для тестов
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        _, mu_test, _, _ = model(test_tensor)
        mu_np = mu_test.cpu().numpy()

    # 4. ГЕНЕРАЦИЯ НОВЫХ СИНТЕТИЧЕСКИХ ОБРАЗЦОВ ИЗ БЕЛОГО ШУМА
    print("\n[Генерация] Создаем 4 абсолютно новых синтетических образца из шума N(0, I)...")
    with torch.no_grad():
        random_latent_codes = torch.randn(4, 2).to(device) # Семплируем случайные точки из N(0, 1)
        synthetic_samples = model.decode(random_latent_codes).cpu().numpy()
    print("✓ Декодер сгенерировал новые физически реалистичные волновые профили!")

    # 5. ПЛАВНАЯ ЛАТЕНТНАЯ ИНТЕРПОЛЯЦИЯ (LATENT SPACE WALK)
    # Выбираем один образец Класса 0 (Резонанс) и один образец Класса 1 (Колокол)
    idx_class0 = np.where(y_test == 0)[0][0]
    idx_class1 = np.where(y_test == 1)[0][0]
    
    z_start = mu_test[idx_class0:idx_class0+1] # Точка в латентном пространстве для Резонанса
    z_end = mu_test[idx_class1:idx_class1+1]   # Точка в латентном пространстве для Колокола

    steps = 8
    alphas = np.linspace(0.0, 1.0, steps)
    interpolated_profiles = []
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1.0 - alpha) * z_start + alpha * z_end
            profile_interp = model.decode(z_interp).cpu().numpy()[0]
            interpolated_profiles.append(profile_interp)

    print(f"✓ Выполнена плавная интерполяция (Морфинг Резонанс -> Колокол) за {steps} шагов!")

    # 6. Визуализация и сохранение графиков
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Панель 1: Непрерывное латентное пространство VAE
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    class_names = ["0: Резонанс", "1: Колокол", "2: Спад"]
    for c_id in range(3):
        mask = (y_test == c_id)
        axes[0, 0].scatter(
            mu_np[mask, 0], mu_np[mask, 1],
            c=colors[c_id], label=class_names[c_id], alpha=0.6, s=35
        )
    # Отображаем линию траектории интерполяции
    z_start_np = z_start.cpu().numpy()[0]
    z_end_np = z_end.cpu().numpy()[0]
    axes[0, 0].plot([z_start_np[0], z_end_np[0]], [z_start_np[1], z_end_np[1]], color="black", lw=2.5, ls="--", label="Траектория морфинга (Latent Walk)")
    axes[0, 0].scatter([z_start_np[0], z_end_np[0]], [z_start_np[1], z_end_np[1]], color="black", s=90, marker="X", zorder=5)

    axes[0, 0].set_title("1. Непрерывное латентное пространство VAE (Smooth Latent Manifold)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Latent Dim Z1")
    axes[0, 0].set_ylabel("Latent Dim Z2")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper right", fontsize=9)

    # Панель 2: Генерация новых профилей из случайного гауссовского шума N(0, I)
    time_pts = np.arange(64)
    gen_colors = ["#8e44ad", "#d35400", "#16a085", "#2c3e50"]
    for i in range(4):
        axes[0, 1].plot(time_pts, synthetic_samples[i], color=gen_colors[i], lw=2.0, label=f"Синтезированный #{i+1} (из N(0,I))")
    axes[0, 1].set_title("2. Генерация новых профилей (Семплирование случайных Z ~ N(0, I))", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Точки профиля")
    axes[0, 1].set_ylabel("Амплитуда")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="upper right", fontsize=9)

    # Панель 3: Плавная интерполяция (Морфинг: Класс 0 -> Класс 1)
    cmap = plt.get_cmap("coolwarm")
    for step_i, prof in enumerate(interpolated_profiles):
        alpha_val = alphas[step_i]
        color_val = cmap(alpha_val)
        label_txt = f"α={alpha_val:.2f} (Старт: Резонанс)" if step_i == 0 else (f"α={alpha_val:.2f} (Финиш: Колокол)" if step_i == steps-1 else None)
        axes[1, 0].plot(time_pts, prof, color=color_val, lw=1.8, label=label_txt)

    axes[1, 0].set_title(f"3. Плавный морфинг сигналов: Резонанс -> Колокол ({steps} шагов интерполяции)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Точки профиля")
    axes[1, 0].set_ylabel("Амплитуда")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="upper right", fontsize=9)

    # Панель 4: Реконструкция одного образца (Факт vs VAE)
    sample_eval_idx = 15
    with torch.no_grad():
        single_input = torch.tensor(X_test[sample_eval_idx:sample_eval_idx+1], dtype=torch.float32).to(device)
        recon_single, _, _, _ = model(single_input)
        recon_np = recon_single.cpu().numpy()[0]

    axes[1, 1].plot(time_pts, X_test[sample_eval_idx], label="Истинный профиль (Ground Truth)", color="black", lw=2.5)
    axes[1, 1].plot(time_pts, recon_np, label="Восстановленный VAE профиль", color="#27ae60", ls="--", lw=2.0)
    eval_mse = np.mean((X_test[sample_eval_idx] - recon_np) ** 2)
    axes[1, 1].set_title(f"4. Точность реконструкции тестового образца (MSE = {eval_mse:.4f})", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Точки профиля")
    axes[1, 1].set_ylabel("Амплитуда")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join("plots", "04_variational_autoencoder.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n📊 График сохранен в файл: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_vae_pipeline()
