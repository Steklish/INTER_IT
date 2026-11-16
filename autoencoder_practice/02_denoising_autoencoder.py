"""
================================================================================
ПРИМЕР 2: DENOISING AUTOENCODER (ОЧИСТКА СИГНАЛОВ И ДАННЫХ ОТ ШУМА)
================================================================================

В этой задаче мы реализуем шумоподавляющий автоэнкодер (Denoising Autoencoder, DAE).
Это один из самых востребованных индустриальных сценариев:
  - Очистка сигналов медицинских датчиков (ЭКГ, ЭЭГ) от наводок и мышечного шума
  - Восстановление зашумленных аудиозаписей и речи (Audio Denoising)
  - Подавление шума в радиочастотных и радиолокационных сигналах (DSP / IoT)

КАК ЭТО РАБОТАЕТ:
В классическом автоэнкодере вход и цель совпадают (X -> X).
В DAE мы искусственно (или естественно) зашумляем вход:
    X_noisy = X_clean + Шум
Сеть принимает X_noisy, сжимает его в латентное пространство и учится восстанавливать
ИДЕАЛЬНО ЧИСТЫЙ ОРИГИНАЛ:
    L = MSE(Decoder(Encoder(X_noisy)), X_clean)

ПОЧЕМУ ЭТО РАБОТАЕТ ЛУЧШЕ ПРОСТЫХ ФИЛЬТРОВ (Moving Average / Low-pass):
Простые частотные фильтры размывают форму сигнала, срезая крутые фронты и полезные пики.
DAE выучивает "анатомию" полезного сигнала (его латентное физическое многообразие).
Любой случайный шум лежит ВНЕ этого многообразия и поэтому проецируется в ноль!
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
# 1. ГЕНЕРАЦИЯ БИОСИГНАЛОВ (ЭКГ-ПОДОБНЫЕ ИМПУЛЬСЫ С ГАРМОНИКАМИ)
# ==============================================================================
def generate_clean_and_noisy_signals(n_samples=2500, signal_len=128, random_seed=42):
    """
    Генерирует сложные волновые импульсы (128 точек):
      - Базовая синусоидальная волна (дыхательный ритм)
      - Острый QRS-подобный пик (сокращение сердца)
      - Вторичный зубец T
    Затем создает 2 версии:
      - X_clean: эталонный идеальный сигнал
      - X_noisy: сигнал, искаженный сильным гауссовским шумом и импульсными наводками
    """
    np.random.seed(random_seed)
    time_steps = np.linspace(0, 1, signal_len)
    
    clean_signals = []
    noisy_signals = []

    for _ in range(n_samples):
        # Случайные параметры формы волны
        freq = np.random.uniform(1.5, 2.5)
        phase = np.random.uniform(0, 2 * np.pi)
        base_wave = 0.5 * np.sin(2 * np.pi * freq * time_steps + phase)

        # Острый импульс (QRS комплекс)
        peak_pos = np.random.uniform(0.35, 0.65)
        peak_width = np.random.uniform(0.02, 0.04)
        peak_amp = np.random.uniform(2.0, 3.5)
        sharp_peak = peak_amp * np.exp(-((time_steps - peak_pos) ** 2) / (2 * peak_width ** 2))

        # Отрицательный зубец S
        s_pos = peak_pos + 0.05
        s_peak = -0.7 * peak_amp * np.exp(-((time_steps - s_pos) ** 2) / (2 * (peak_width * 0.8) ** 2))

        clean_sig = base_wave + sharp_peak + s_peak

        # 1. Сильный гауссовский шум (наводка электросети)
        gaussian_noise = np.random.normal(0, 0.45, size=signal_len)
        
        # 2. Импульсные выпадения / всплески (Salt-and-Pepper)
        impulse_noise = np.zeros(signal_len)
        num_spikes = np.random.randint(1, 4)
        spike_idx = np.random.choice(signal_len, size=num_spikes, replace=False)
        impulse_noise[spike_idx] = np.random.choice([-1.8, 1.8], size=num_spikes)

        noisy_sig = clean_sig + gaussian_noise + impulse_noise

        clean_signals.append(clean_sig)
        noisy_signals.append(noisy_sig)

    return np.array(clean_signals, dtype=np.float32), np.array(noisy_signals, dtype=np.float32)


# ==============================================================================
# 2. АРХИТЕКТУРА 1D СВЕРТОЧНОГО ШУМОПОДАВЛЯЮЩЕГО АВТОЭНКОДЕРА (CONV1D DAE)
# ==============================================================================
class Conv1DDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Входная размерность: (batch, 1 канал, 128 точек)

        # 1. Энкодер: Свертки с уменьшением длины через Stride=2
        # 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2), # 128 -> 64
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2), # 64 -> 32
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        # 2. Декодер: Транспонированные свертки (ConvTranspose1d) для точного восстановления
        # 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # 16 -> 32
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1), # 32 -> 64
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1)    # 64 -> 128
        )

    def forward(self, x):
        # x: (batch, 1, 128)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


# ==============================================================================
# 3. МЕТРИКА ОЦЕНКИ КАЧЕСТВА: SIGNAL-TO-NOISE RATIO (SNR В ДЕЦИБЕЛАХ)
# ==============================================================================
def calculate_snr_db(clean, processed):
    """
    Вычисляет отношение сигнал/шум в dB:
      SNR = 10 * log10( P_signal / P_noise )
    Чем выше значение в dB, тем чище сигнал!
    """
    signal_power = np.sum(clean ** 2, axis=1)
    noise_power = np.sum((clean - processed) ** 2, axis=1) + 1e-8
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return np.mean(snr_db)


# ==============================================================================
# 4. ОСНОВНОЙ ПАЙПЛАЙН: ОБУЧЕНИЕ И АНАЛИЗ ОЧИСТКИ
# ==============================================================================
def run_denoising_pipeline():
    print("=" * 70)
    print("🧹 2. DENOISING AUTOENCODER: ФИЛЬТРАЦИЯ СЛОЖНЫХ СИГНАЛОВ")
    print("=" * 70)

    # 1. Генерация сигналов
    X_clean, X_noisy = generate_clean_and_noisy_signals(n_samples=2500, signal_len=128, random_seed=42)
    
    # Train / Test split
    X_clean_train, X_clean_test, X_noisy_train, X_noisy_test = train_test_split(
        X_clean, X_noisy, test_size=0.2, random_state=42
    )
    print(f"✓ Сгенерировано 2500 волновых сигналов длины 128 точек")
    print(f"✓ Train: {len(X_clean_train)}, Test: {len(X_clean_test)}")

    # 2. Исходное качество зашумленного сигнала (до очистки)
    snr_initial = calculate_snr_db(X_clean_test, X_noisy_test)
    mse_initial = np.mean((X_clean_test - X_noisy_test) ** 2)
    print(f"\n📊 До обработки (Зашумленный вход):")
    print(f"   - Исходный SNR:  {snr_initial:.2f} dB")
    print(f"   - Исходный MSE:  {mse_initial:.4f}")

    # 3. Бейзлайн: Простое скользящее среднее (Moving Average Filter с окном 5)
    def moving_average_filter(signals, window=5):
        filtered = np.zeros_like(signals)
        pad = window // 2
        for i in range(len(signals)):
            padded = np.pad(signals[i], pad, mode="edge")
            filtered[i] = np.convolve(padded, np.ones(window)/window, mode="valid")
        return filtered

    X_test_ma_filtered = moving_average_filter(X_noisy_test, window=5)
    snr_ma = calculate_snr_db(X_clean_test, X_test_ma_filtered)
    mse_ma = np.mean((X_clean_test - X_test_ma_filtered) ** 2)
    print(f"\n[Классический бейзлайн] Скользящий средний фильтр (Window=5):")
    print(f"   - Фильтрованный SNR: {snr_ma:.2f} dB (Прирост: +{snr_ma - snr_initial:.2f} dB)")
    print(f"   - Фильтрованный MSE: {mse_ma:.4f}")

    # 4. Обучение нейросетевого Conv1D Denoising Autoencoder
    print("\n[Нейросеть] Обучаем Conv1D Denoising Autoencoder на GPU/CPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Используется устройство: {device}")

    model = Conv1DDenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

    # Подготовка PyTorch DataLoader: Вход = X_noisy, Таргет = X_clean!
    # Добавляем измерение канала: (N, 1, 128)
    train_noisy_tensor = torch.tensor(X_noisy_train[:, None, :], dtype=torch.float32)
    train_clean_tensor = torch.tensor(X_clean_train[:, None, :], dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_noisy_tensor, train_clean_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    epochs = 35
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_noisy, batch_clean in train_loader:
            batch_noisy = batch_noisy.to(device)
            batch_clean = batch_clean.to(device)

            optimizer.zero_grad()
            denoised_output = model(batch_noisy)
            loss = criterion(denoised_output, batch_clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_noisy)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Эпоха {epoch+1:2d}/{epochs:2d} | Функция потерь DAE (MSE): {epoch_loss/len(X_clean_train):.5f}")

    # 5. Тестирование модели DAE на чистом тесте
    model.eval()
    with torch.no_grad():
        test_noisy_tensor = torch.tensor(X_noisy_test[:, None, :], dtype=torch.float32).to(device)
        denoised_tensor = model(test_noisy_tensor)
        X_test_dae_denoised = denoised_tensor.cpu().numpy()[:, 0, :]

    snr_dae = calculate_snr_db(X_clean_test, X_test_dae_denoised)
    mse_dae = np.mean((X_clean_test - X_test_dae_denoised) ** 2)

    print("\n" + "=" * 65)
    print(f"{'Метод фильтрации':<25} | {'SNR (dB)':<10} | {'MSE Ошибка':<12} | {'Прирост SNR':<12}")
    print("-" * 65)
    print(f"{'1. Без обработки (Noisy)':<25} | {snr_initial:8.2f} dB | {mse_initial:12.4f} | {'База (0 dB)':<12}")
    print(f"{'2. Moving Average Filter':<25} | {snr_ma:8.2f} dB | {mse_ma:12.4f} | {f'+{snr_ma-snr_initial:.2f} dB':<12}")
    print(f"{'3. Conv1D Autoencoder':<25}   | {snr_dae:8.2f} dB | {mse_dae:12.4f} | {f'+{snr_dae-snr_initial:.2f} dB':<12}")
    print("=" * 65)
    print(f"🏆 Превосходство DAE: ошибка MSE снижена на {((mse_initial - mse_dae)/mse_initial)*100:.1f}%, сигнал усилен на +{snr_dae-snr_initial:.1f} dB!")

    # 6. Визуализация результатов
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False)

    # График 1: Подробный срез одного образца (Чистый vs Зашумленный vs DAE vs MA)
    idx = 7
    time_pts = np.arange(128)
    axes[0].plot(time_pts, X_clean_test[idx], label="1. Истинный чистый сигнал (Clean Ground Truth)", color="black", lw=2.5)
    axes[0].plot(time_pts, X_noisy_test[idx], label="2. Зашумленный вход (Noisy Input)", color="#e74c3c", alpha=0.5, lw=1.2)
    axes[0].plot(time_pts, X_test_ma_filtered[idx], label="3. Скользящее среднее (Срезает острый пик!)", color="#e67e22", ls="--", lw=1.8)
    axes[0].plot(time_pts, X_test_dae_denoised[idx], label=f"4. Восстановление DAE (SNR={snr_dae:.1f} dB)", color="#27ae60", lw=2.2)
    axes[0].set_title(f"Сравнение фильтрации для образца #{idx} (Обратите внимание, как DAE сохраняет острый пик QRS)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Амплитуда", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    # График 2: Разность ошибки (Residual Noise = Clean - Processed)
    axes[1].plot(time_pts, X_noisy_test[idx] - X_clean_test[idx], label="Остаточный шум входа (Noise)", color="#e74c3c", alpha=0.4)
    axes[1].plot(time_pts, X_test_dae_denoised[idx] - X_clean_test[idx], label="Остаточная ошибка DAE (DAE Error)", color="#27ae60", lw=1.8)
    axes[1].axhline(0, color="black", ls=":", alpha=0.5)
    axes[1].set_title("Профиль остаточного шума после очистки (DAE практически обнуляет помехи)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Ошибка (Δ)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    # График 3: Распределение прироста SNR по всем 500 тестовым сигналам
    snr_gains_dae = 10.0 * np.log10(np.sum(X_clean_test ** 2, axis=1) / (np.sum((X_clean_test - X_test_dae_denoised) ** 2, axis=1) + 1e-8)) - \
                    10.0 * np.log10(np.sum(X_clean_test ** 2, axis=1) / (np.sum((X_clean_test - X_noisy_test) ** 2, axis=1) + 1e-8))
    
    snr_gains_ma = 10.0 * np.log10(np.sum(X_clean_test ** 2, axis=1) / (np.sum((X_clean_test - X_test_ma_filtered) ** 2, axis=1) + 1e-8)) - \
                   10.0 * np.log10(np.sum(X_clean_test ** 2, axis=1) / (np.sum((X_clean_test - X_noisy_test) ** 2, axis=1) + 1e-8))

    axes[2].hist(snr_gains_ma, bins=30, alpha=0.5, color="#e67e22", label=f"Moving Average Filter (Средний прирост: +{np.mean(snr_gains_ma):.1f} dB)")
    axes[2].hist(snr_gains_dae, bins=30, alpha=0.6, color="#27ae60", label=f"Conv1D Denoising Autoencoder (Средний прирост: +{np.mean(snr_gains_dae):.1f} dB)")
    axes[2].set_title("Распределение прироста SNR (Качество подавления шума на всех 500 тестах)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Прирост SNR в децибелах (dB, чем больше, тем лучше)")
    axes[2].set_ylabel("Количество образцов")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plot_path = os.path.join("plots", "02_denoising_autoencoder.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n📊 График сохранен в файл: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_denoising_pipeline()
