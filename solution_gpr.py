"""
Решение на основе Геостатистики / Гауссовских процессов (Kriging / GPR).
Использует пространственно-временное конвективное ядро:
- Координаты: Лагранжевы смещения (xi, eta, zeta) относительно центра шлейфа.
- Модель: GaussianProcessRegressor (Matérn + WhiteKernel).
- 90%-й доверительный коридор: вычисляется аналитически из апостериорной дисперсии sigma!
- Пост-процессинг: 3D численное интегрирование массы и поиск прорыва на водозаборе.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from solution import HydroFeatureBuilder

def run_gpr_solution(data_dir="data", output_csv="submission_gpr.csv", output_meta="submission_gpr_meta.json"):
    print("=" * 65, flush=True)
    print("    РЕШЕНИЕ 1: ГЕОСТАТИСТИКА И ГАУССОВСКИЕ ПРОЦЕССЫ (GPR / KRIGING)    ", flush=True)
    print("=" * 65, flush=True)
    
    fb = HydroFeatureBuilder(data_dir=data_dir)
    conc_df = pd.read_csv(os.path.join(data_dir, "concentration.csv"))
    
    print(f"Загрузка {len(conc_df)} замеров...", flush=True)
    X_train_df = fb.build_features(conc_df)
    y_train = conc_df["concentration"].values.astype(np.float32)
    
    # Ключевые признаки: Лагранжевы координаты (xi, eta, zeta) + время + c_phys
    feature_cols = ["xi_lagrange", "eta_lagrange", "zeta_lagrange", "t_days", "c_phys"]
    X_tr = X_train_df[feature_cols].values
    
    scaler_mean = np.mean(X_tr, axis=0)
    scaler_std = np.std(X_tr, axis=0) + 1e-5
    X_tr_norm = (X_tr - scaler_mean) / scaler_std
    y_tr_log = np.log1p(np.maximum(0.0, y_train))
    
    # Отбор репрезентативных точек: все положительные замеры + часть нулевых (всего 400 точек для высокой скорости)
    np.random.seed(42)
    pos_idx = np.where(y_train > 0.05)[0]
    zero_idx = np.where(y_train <= 0.05)[0]
    sub_zero = np.random.choice(zero_idx, size=min(len(zero_idx), 200), replace=False)
    selected_idx = np.concatenate([pos_idx, sub_zero])
    
    print(f"Выборка для GPR: {len(selected_idx)} точек ({len(pos_idx)} положительных + {len(sub_zero)} фоновых)...", flush=True)
    
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(length_scale=np.ones(len(feature_cols)), nu=1.5) + WhiteKernel(noise_level=0.05)
    
    print("Обучение Гауссовского процесса...", flush=True)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,
        n_restarts_optimizer=0,
        random_state=42,
        normalize_y=True
    )
    gpr.fit(X_tr_norm[selected_idx], y_tr_log[selected_idx])
    print(f"Ядро GPR настроено: {gpr.kernel_}", flush=True)
    
    # Прогноз на тестовых запросах (queries_test.csv)
    queries_df = pd.read_csv(os.path.join(data_dir, "queries_test.csv"))
    print(f"Инференс для {len(queries_df)} тестовых запросов...", flush=True)
    X_test_df = fb.build_features(queries_df)
    X_te_norm = (X_test_df[feature_cols].values - scaler_mean) / scaler_std
    
    mu_log, std_log = gpr.predict(X_te_norm, return_std=True)
    c_pred = np.expm1(np.maximum(0.0, mu_log))
    
    # Аналитический 90%-й доверительный коридор из апостериорной неопределенности:
    raw_lower = np.expm1(np.maximum(0.0, mu_log - 1.645 * std_log))
    raw_upper = np.expm1(np.maximum(0.0, mu_log + 1.645 * std_log))
    
    c_lower = np.maximum(0.0, raw_lower)
    c_upper = np.maximum(c_lower + 0.1, raw_upper)
    
    sub_df = pd.DataFrame({
        "query_id": queries_df["query_id"],
        "c_pred": np.round(c_pred, 3),
        "c_lower": np.round(c_lower, 3),
        "c_upper": np.round(c_upper, 3)
    })
    sub_df.to_csv(output_csv, index=False)
    print(f"Предсказания сохранены в {output_csv}", flush=True)
    
    # -------------------------------------------------------------
    # Расчет массы на 01.01.2026 и времени прорыва
    # -------------------------------------------------------------
    print("Интегрирование массы на 01.01.2026...", flush=True)
    date_target = datetime(2026, 1, 1)
    gx = np.arange(50, 5000, 100.0)
    gy = np.arange(1200, 3800, 100.0)
    gz = np.arange(-45.0, 0.0, 10.0)
    dV = 100.0 * 100.0 * 10.0
    
    mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_df = pd.DataFrame({
        "x": mesh_x.ravel(), "y": mesh_y.ravel(), "z": mesh_z.ravel(),
        "date": date_target.strftime("%Y-%m-%d")
    })
    X_grid_df = fb.build_features(grid_df)
    X_grid_norm = (X_grid_df[feature_cols].values - scaler_mean) / scaler_std
    
    p_log = gpr.predict(X_grid_norm)
    c_grid = np.expm1(np.maximum(0.0, p_log))
    c_clean = np.where(c_grid >= 0.01, c_grid, 0.0)
    gpr_mass = float(np.sum(c_clean * 1e-6 * fb.mean_porosity * dV))
    print(f"Масса TCE на 01.01.2026 (GPR): {gpr_mass:.2f} кг", flush=True)
    
    # Время прорыва на Заречном водозаборе
    print("Скан кривой прорыва на Заречном водозаборе...", flush=True)
    dates_scan = [fb.sim_start + timedelta(days=d) for d in range(600, 365 * 6, 3)]
    zar_df = pd.DataFrame({
        "x": np.full(len(dates_scan), fb.zar_x),
        "y": np.full(len(dates_scan), fb.zar_y),
        "z": np.full(len(dates_scan), fb.zar_z),
        "date": [d.strftime("%Y-%m-%d") for d in dates_scan]
    })
    X_zar_df = fb.build_features(zar_df)
    X_zar_norm = (X_zar_df[feature_cols].values - scaler_mean) / scaler_std
    zar_pred = np.expm1(np.maximum(0.0, gpr.predict(X_zar_norm)))
    zar_smooth = pd.Series(zar_pred).rolling(window=3, min_periods=1).mean().values
    
    gpr_arrival = None
    for i, val in enumerate(zar_smooth):
        if val >= 5.0:
            gpr_arrival = dates_scan[i].strftime("%Y-%m-%d")
            break
    if gpr_arrival is None:
        gpr_arrival = dates_scan[np.argmax(zar_smooth)].strftime("%Y-%m-%d")
        
    print(f"Дата прорыва на Заречном (GPR): {gpr_arrival}", flush=True)
    
    meta = {
        "mass_2026": round(gpr_mass, 2),
        "arrival_time": gpr_arrival
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("GPR решение успешно завершено!", flush=True)

if __name__ == "__main__":
    run_gpr_solution()
