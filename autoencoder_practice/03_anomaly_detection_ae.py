"""
================================================================================
ПРИМЕР 3: ДЕТЕКЦИЯ АНОМАЛИЙ ПО ОШИБКЕ РЕКОНСТРУКЦИИ (RECONSTRUCTION ERROR)
================================================================================

В этой задаче мы изучаем важнейший промышленный паттерн:
Semi-Supervised / One-Class детекция аномалий и предиктивное обслуживание (Predictive Maintenance).

ПРОБЛЕМА РЕАЛЬНОГО МИРА:
На реальном производстве (турбины, реакторы, банковские транзакции, серверные фермы)
аварии случаются крайне редко (0.01% случаев).
У нас нет размеченного датасета с тысячами примеров всех возможных катастроф.
Зато у нас есть миллионы записей ШТАТНОЙ НОРМАЛЬНОЙ РАБОТЫ оборудования.

КАК АВТОЭНКОДЕР РЕШАЕТ ЭТУ ЗАДАЧУ:
1. Сеть обучается ИСКЛЮЧИТЕЛЬНО на нормальных данных (X_train содержит 0% аномалий).
2. Автоэнкодер выучивает законы физики нормального режима: как давление связано
   с оборотами ротора, а температура масла — с вибрацией подшипников.
3. На этапе инференса мы вычисляем Ошибку Реконструкции (Reconstruction Error):
       Anomaly_Score(x) = MSE(x, Decoder(Encoder(x)))
4. Если пришел нормальный режим -> сеть легко сжимает и восстанавливает его -> MSE низкий.
5. Если произошла авария (разбалансировка вала, перегрев) -> сломались выученные
   корреляции между датчиками -> сеть "не понимает" этот паттерн -> MSE взлетает выше порога tau!

БОНУС: ЛОКАЛИЗАЦИЯ ПЕРВОПРИЧИНЫ (ROOT CAUSE ANALYSIS):
Поскольку автоэнкодер восстанавливает каждый сенсор отдельно, мы можем посмотреть,
какой именно датчик дал максимальную ошибку |x_i - x̂_i|, и сразу указать инженеру,
в каком узле возник дефект!
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, confusion_matrix


# ==============================================================================
# 1. ГЕНЕРАЦИЯ ТЕЛЕМЕТРИИ ТУРБИНЫ (16 ДАТЧИКОВ: НОРМА И 3 ВИДА АВАРИЙ)
# ==============================================================================
def generate_turbine_telemetry(n_normal=4000, n_anomaly=500, random_seed=42):
    """
    Генерирует 16 каналов телеметрии газовой турбины:
      - Датчики 0..3: Вибрация подшипников (X, Y, Z, Корпус)
      - Датчики 4..7: Температуры (Масло, Входные газы, Камера сгорания, Выхлоп)
      - Датчики 8..11: Давления (Вход воздуха, Компрессор, Топливная рампа, Смазка)
      - Датчики 12..15: Электромеханика (Обороты RPM, Мощность, Ток статора, Поток)

    Аномалии:
      1. Дисбаланс ротора (Rotor Imbalance): резкий скачок вибрации (датчики 0..3)
      2. Перегрев подшипников (Thermal Overheating): аномальная температура масла (датчик 4)
      3. Кавитация / Срыв компрессора (Pressure Surge): броски давлений (датчики 8..10)
    """
    np.random.seed(random_seed)
    n_features = 16

    def generate_normal_batch(n):
        # Базовая нагрузка от 0.6 до 1.0
        load = np.random.uniform(0.6, 1.0, size=(n, 1))
        # Физические корреляции между параметрами
        rpm = 3000.0 * load + np.random.normal(0, 15, size=(n, 1))
        power = 50.0 * (load ** 1.5) + np.random.normal(0, 0.5, size=(n, 1))
        temp_exhaust = 450.0 + 200.0 * load + np.random.normal(0, 5, size=(n, 1))
        temp_oil = 65.0 + 25.0 * load + np.random.normal(0, 2, size=(n, 1))
        vib_base = 1.2 + 0.8 * load + np.random.normal(0, 0.1, size=(n, 4))
        pressures = 15.0 * load + np.random.normal(0, 0.3, size=(n, 4))
        other_temps = 300.0 * load + np.random.normal(0, 8, size=(n, 2))
        current = 120.0 * load + np.random.normal(0, 2, size=(n, 2))

        # Собираем 16-мерный вектор
        X = np.hstack([vib_base, temp_oil, other_temps, temp_exhaust, pressures, rpm, power, current])
        return X

    # 1. Нормальные данные (для обучения и теста)
    X_normal = generate_normal_batch(n_normal)

    # 2. Аномальные данные (только для тестирования!)
    X_anom_1 = generate_normal_batch(n_anomaly // 3)
    # Авария 1: Дисбаланс ротора (вибрация подшипников x4..x7)
    X_anom_1[:, 0:4] += np.random.uniform(5.0, 9.0, size=(len(X_anom_1), 4))

    X_anom_2 = generate_normal_batch(n_anomaly // 3)
    # Авария 2: Перегрев масла (датчик 4 взлетает до 130°C)
    X_anom_2[:, 4] += np.random.uniform(40.0, 65.0, size=len(X_anom_2))

    X_anom_3 = generate_normal_batch(n_anomaly - 2 * (n_anomaly // 3))
    # Авария 3: Срыв давления компрессора (датчики 8..11 падают или пульсируют)
    X_anom_3[:, 8:11] -= np.random.uniform(7.0, 11.0, size=(len(X_anom_3), 3))

    X_anomalies = np.vstack([X_anom_1, X_anom_2, X_anom_3])
    anom_types = ["Дисбаланс ротора"] * len(X_anom_1) + \
                 ["Перегрев масла"] * len(X_anom_2) + \
                 ["Срыв компрессора"] * len(X_anom_3)

    return X_normal.astype(np.float32), X_anomalies.astype(np.float32), anom_types


# ==============================================================================
# 2. АРХИТЕКТУРА АВТОЭНКОДЕРА ДЛЯ СЕНСОРНОЙ ТЕЛЕМЕТРИИ
# ==============================================================================
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim=16, latent_dim=4):
        super().__init__()
        # Энкодер сжимает 16 -> 10 -> 6 -> 4 (латентное ядро)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.GELU(),
            nn.Linear(10, 6),
            nn.GELU(),
            nn.Linear(6, latent_dim)
        )
        # Декодер восстанавливает 4 -> 6 -> 10 -> 16
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 6),
            nn.GELU(),
            nn.Linear(6, 10),
            nn.GELU(),
            nn.Linear(10, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# ==============================================================================
# 3. ОСНОВНОЙ ПАЙПЛАЙН: ОБУЧЕНИЕ НА НОРМЕ, ДЕТЕКЦИЯ И ROOT CAUSE
# ==============================================================================
def run_anomaly_detection_ae_pipeline():
    print("=" * 70)
    print("🛡️ 3. ДЕТЕКЦИЯ АНОМАЛИЙ НА БАЗЕ ОШИБКИ РЕКОНСТРУКЦИИ АВТОЭНКОДЕРА")
    print("=" * 70)

    # 1. Генерация данных
    X_norm, X_anom, anom_types = generate_turbine_telemetry(n_normal=4000, n_anomaly=600, random_seed=42)

    # Нормальные данные делим на Обучение (80%) и Валидацию нормы (20%)
    n_train = 3200
    X_train_normal = X_norm[:n_train]
    X_val_normal = X_norm[n_train:]

    # Стандартизация (обучаем Scaler ТОЛЬКО на нормальных данных!)
    norm_mean = np.mean(X_train_normal, axis=0)
    norm_std = np.std(X_train_normal, axis=0) + 1e-6

    X_train_scaled = (X_train_normal - norm_mean) / norm_std
    X_val_scaled = (X_val_normal - norm_mean) / norm_std
    X_anom_scaled = (X_anom - norm_mean) / norm_std

    print(f"✓ Обучающая выборка: {len(X_train_scaled)} нормальных записей (0% аномалий)")
    print(f"✓ Тестовая выборка:  {len(X_val_scaled)} нормальных + {len(X_anom_scaled)} аварийных записей")

    # 2. Обучение автоэнкодера (PyTorch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Вычисления на устройстве: {device}")

    model = AnomalyAutoencoder(input_dim=16, latent_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-5)

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=64, shuffle=True)

    epochs = 35
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x = model(batch_x)
            loss = criterion(recon_x, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Эпоха {epoch+1:2d}/{epochs:2d} | Функция потерь реконструкции: {epoch_loss/len(X_train_scaled):.5f}")

    # 3. Калибровка порога аномальности на валидационной НОРМАЛЬНОЙ выборке
    model.eval()
    with torch.no_grad():
        val_recon = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        val_mse = np.mean((X_val_scaled - val_recon) ** 2, axis=1)

        # Вычисляем порог tau как 99.5-й перцентиль ошибок нормы
        tau_threshold = np.percentile(val_mse, 99.5)
        print(f"\n✓ Калиброванный порог нормы (99.5-й перцентиль): tau = {tau_threshold:.4f}")

    # 4. Тестирование на смеси нормы и аварийных сценариев
    X_test_combined = np.vstack([X_val_scaled, X_anom_scaled])
    y_test_combined = np.array([0] * len(X_val_scaled) + [1] * len(X_anom_scaled))

    with torch.no_grad():
        test_recon = model(torch.tensor(X_test_combined, dtype=torch.float32).to(device)).cpu().numpy()
        # Ошибка каждого сенсора и средняя ошибка образца
        channel_errors = (X_test_combined - test_recon) ** 2
        test_mse = np.mean(channel_errors, axis=1)

    # Предсказание класса: 1 если MSE > tau, иначе 0
    y_pred_binary = (test_mse > tau_threshold).astype(int)

    # 5. Метрики качества детекции
    roc_auc = roc_auc_score(y_test_combined, test_mse)
    prec = precision_score(y_test_combined, y_pred_binary)
    rec = recall_score(y_test_combined, y_pred_binary)
    f1 = f1_score(y_test_combined, y_pred_binary)
    cm = confusion_matrix(y_test_combined, y_pred_binary)

    print("\n" + "=" * 65)
    print("📈 РЕЗУЛЬТАТЫ ДЕТЕКЦИИ АНОМАЛИЙ НА ТЕСТЕ:")
    print("-" * 65)
    print(f"★ ROC-AUC Score:      {roc_auc:.4f} (Идеальная разделимость!)")
    print(f"★ Precision (Точность): {prec*100:.2f}%")
    print(f"★ Recall (Полнота):     {rec*100:.2f}%")
    print(f"★ F1-Score:            {f1*100:.2f}%")
    print("-" * 65)
    print(f"Матрица ошибок (Confusion Matrix):")
    print(f"   [Истинная норма : {cm[0,0]} верно | {cm[0,1]} ложных тревог]")
    print(f"   [Истинные аварии: {cm[1,1]} поймано | {cm[1,0]} пропущено]")
    print("=" * 65)

    # 6. Root Cause Analysis: Какой сенсор дал сбой при разных авариях?
    sensor_names = [
        "Vib_X", "Vib_Y", "Vib_Z", "Vib_Body",
        "Temp_Oil", "Temp_Inlet", "Temp_Combust", "Temp_Exhaust",
        "Press_Air", "Press_Comp", "Press_Fuel", "Press_Lube",
        "RPM", "Power", "Current_A", "Current_B"
    ]
    anom_channel_errors = channel_errors[len(X_val_scaled):]
    top_sensor_per_anom = np.argmax(anom_channel_errors, axis=1)

    print("\n🔍 АНАЛИЗ ПЕРВОПРИЧИНЫ (ROOT CAUSE ANALYSIS):")
    for anom_name, start_idx, end_idx in [
        ("Дисбаланс ротора", 0, 200),
        ("Перегрев масла", 200, 400),
        ("Срыв компрессора", 400, 600)
    ]:
        top_sensors = [sensor_names[i] for i in top_sensor_per_anom[start_idx:end_idx]]
        most_common = max(set(top_sensors), key=top_sensors.count)
        print(f"• Авария '{anom_name}': автоэнкодер выделил датчик [{most_common}] как главный источник аномальной ошибки!")

    # 7. Визуализация и сохранение дашборда
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Панель 1: Временная шкала ошибок реконструкции с порогом
    time_axis = np.arange(len(test_mse))
    axes[0, 0].plot(time_axis[:len(X_val_scaled)], test_mse[:len(X_val_scaled)], color="#2ecc71", alpha=0.7, label="Нормальный режим (MSE < tau)")
    
    # 3 зоны аварий
    s_idx = len(X_val_scaled)
    axes[0, 0].plot(time_axis[s_idx:s_idx+200], test_mse[s_idx:s_idx+200], color="#e74c3c", label="Авария 1: Дисбаланс ротора")
    axes[0, 0].plot(time_axis[s_idx+200:s_idx+400], test_mse[s_idx+200:s_idx+400], color="#e67e22", label="Авария 2: Перегрев масла")
    axes[0, 0].plot(time_axis[s_idx+400:], test_mse[s_idx+400:], color="#9b59b6", label="Авария 3: Срыв компрессора")
    
    axes[0, 0].axhline(tau_threshold, color="black", linestyle="--", lw=2, label=f"Порог тревоги tau = {tau_threshold:.2f}")
    axes[0, 0].set_title("1. Ошибка реконструкции автоэнкодера (Anomaly Score во времени)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Reconstruction Error (MSE)")
    axes[0, 0].set_xlabel("Номер тестового замера")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper left", fontsize=8)

    # Панель 2: Плотность распределения ошибок (Normal vs Anomalies)
    axes[0, 1].hist(test_mse[:len(X_val_scaled)], bins=30, alpha=0.6, color="#2ecc71", density=True, label="Нормальный режим")
    axes[0, 1].hist(test_mse[len(X_val_scaled):], bins=30, alpha=0.6, color="#e74c3c", density=True, label="Аварийные режимы")
    axes[0, 1].axvline(tau_threshold, color="black", linestyle="--", lw=2, label=f"Порог tau = {tau_threshold:.2f}")
    axes[0, 1].set_title(f"2. Разделимость классов (Плотность распределения MSE)\nROC-AUC = {roc_auc:.4f}", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("MSE реконструкции")
    axes[0, 1].set_ylabel("Плотность вероятности")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="upper right", fontsize=9)

    # Панель 3: Кривая ROC-AUC
    fpr, tpr, _ = roc_curve(y_test_combined, test_mse)
    axes[1, 0].plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC кривая (AUC = {roc_auc:.4f})")
    axes[1, 0].plot([0, 1], [0, 1], color="gray", linestyle=":")
    axes[1, 0].set_title("3. ROC-кривая качества детекции аномалий", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("False Positive Rate (Ложные тревоги)")
    axes[1, 0].set_ylabel("True Positive Rate (Полнота обнаружения)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="lower right", fontsize=9)

    # Панель 4: Вклад каждого датчика в ошибку реконструкции (Root Cause Heatmap)
    # Средняя ошибка каждого датчика для нормы и каждого из 3 типов аварий
    mean_errors_matrix = np.array([
        np.mean(channel_errors[:len(X_val_scaled)], axis=0),
        np.mean(channel_errors[s_idx:s_idx+200], axis=0),
        np.mean(channel_errors[s_idx+200:s_idx+400], axis=0),
        np.mean(channel_errors[s_idx+400:], axis=0)
    ])
    
    im = axes[1, 1].imshow(mean_errors_matrix, cmap="YlOrRd", aspect="auto")
    axes[1, 1].set_title("4. Локализация первопричины (Вклад каждого сенсора в ошибку)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(np.arange(16))
    axes[1, 1].set_xticklabels(sensor_names, rotation=60, ha="right", fontsize=8)
    axes[1, 1].set_yticks(np.arange(4))
    axes[1, 1].set_yticklabels(["Норма", "Дисбаланс ротора", "Перегрев масла", "Срыв компрессора"], fontsize=9)
    fig.colorbar(im, ax=axes[1, 1], orientation="vertical", label="MSE ошибки сенсора")

    plt.tight_layout()
    plot_path = os.path.join("plots", "03_anomaly_detection_ae.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n📊 График сохранен в файл: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_anomaly_detection_ae_pipeline()
