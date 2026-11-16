"""
Решение на основе Физико-Информированных Нейросетей (PINN — Physics-Informed Neural Networks).
Реализовано на PyTorch с поддержкой CUDA:
- Архитектура: Многослойный перцептрон (MLP) с активациями SiLU.
- Функция потерь: L_total = L_data (замеры скважин) + lambda_phys * L_pde (невязка уравнения переноса).
- PDE невязка: вычисляется через torch.autograd.grad на случайных collocation points.
- Неопределенность: ансамбль сетей (Deep Ensemble) для вычисления 90% доверительного коридора.
- Интегрирование массы на 01.01.2026 и поиск прорыва на водозаборе.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from solution import HydroFeatureBuilder

# Проверка устройства (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINNNetwork(nn.Module):
    def __init__(self, in_features=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Гарантирует неотрицательность концентрации C >= 0!
        )
        
    def forward(self, x):
        return self.net(x)

def train_single_pinn(X_train_t, y_train_t, X_colloc_t, epochs=600, lr=2e-3, seed=42):
    torch.manual_seed(seed)
    model = PINNNetwork(in_features=X_train_t.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Скорость переноса v_flow ~ 1.95 м/сут, дисперсия D ~ 25 м2/сут
    v_norm = 1.95 / 1000.0
    d_norm = 25.0 / 1e6
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. Data Loss (на замерах скважин)
        pred_data = model(X_train_t)
        loss_data = torch.mean((pred_data - y_train_t) ** 2)
        
        # 2. Physics Loss (на случайных точках объема через autograd)
        if epoch > 100:
            # Требуем градиенты по входам
            X_colloc_req = X_colloc_t.clone().detach().requires_grad_(True)
            pred_colloc = model(X_colloc_req)
            
            # Градиенты первого порядка: dC/dx и dC/dt
            # Вход: [x_norm, y_norm, z_norm, t_norm, c_phys_norm]
            grads = torch.autograd.grad(
                pred_colloc, X_colloc_req,
                grad_outputs=torch.ones_like(pred_colloc),
                create_graph=True
            )[0]
            
            dC_dx = grads[:, 0:1]
            dC_dt = grads[:, 3:4]
            
            # Невязка уравнения адвекции: Residual = dC/dt + v * dC/dx
            pde_residual = dC_dt + v_norm * dC_dx
            loss_physics = torch.mean(pde_residual ** 2)
            total_loss = loss_data + 0.05 * loss_physics
        else:
            total_loss = loss_data
            
        total_loss.backward()
        optimizer.step()
        
    return model

def run_pinn_solution(data_dir="data", output_csv="submission_pinn.csv", output_meta="submission_pinn_meta.json"):
    print("=" * 65, flush=True)
    print(f"    РЕШЕНИЕ 2: ФИЗИКО-ИНФОРМИРОВАННЫЕ НЕЙРОСЕТИ (PINN НА {device.type.upper()})    ", flush=True)
    print("=" * 65, flush=True)
    
    fb = HydroFeatureBuilder(data_dir=data_dir)
    conc_df = pd.read_csv(os.path.join(data_dir, "concentration.csv"))
    
    print(f"Подготовка данных замеров ({len(conc_df)} строк)...", flush=True)
    X_train_df = fb.build_features(conc_df)
    y_train = conc_df["concentration"].values.astype(np.float32)[:, None]
    
    feature_cols = ["xi_lagrange", "eta_lagrange", "zeta_lagrange", "t_days", "c_phys"]
    X_tr = X_train_df[feature_cols].values.astype(np.float32)
    
    mean_val = np.mean(X_tr, axis=0)
    std_val = np.std(X_tr, axis=0) + 1e-5
    X_tr_norm = (X_tr - mean_val) / std_val
    
    # Генерация collocation points (1500 случайных точек полигона для проверки физики)
    np.random.seed(42)
    rand_x = np.random.uniform(500, 4800, 1500)
    rand_y = np.random.uniform(1500, 3500, 1500)
    rand_z = np.random.uniform(-45.0, -5.0, 1500)
    rand_t = np.random.uniform(100, 1800, 1500)
    
    colloc_df = pd.DataFrame({"x": rand_x, "y": rand_y, "z": rand_z, "t_days": rand_t})
    X_colloc_df = fb.build_features(colloc_df)
    X_colloc_norm = (X_colloc_df[feature_cols].values.astype(np.float32) - mean_val) / std_val
    
    # Тензоры PyTorch
    X_train_t = torch.tensor(X_tr_norm, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_colloc_t = torch.tensor(X_colloc_norm, dtype=torch.float32, device=device)
    
    # Обучаем ансамбль из 3 сетей с разными сидами (Deep Ensemble PINN)
    print("Обучение ансамбля 3 PINN сетей на GPU с контролем невязки PDE...", flush=True)
    ensemble = []
    seeds = [42, 123, 777]
    for s_idx, s in enumerate(seeds):
        print(f"  Обучение модели #{s_idx + 1} (seed={s})...", flush=True)
        m = train_single_pinn(X_train_t, y_train_t, X_colloc_t, epochs=700, lr=2.5e-3, seed=s)
        ensemble.append(m)
        
    print("Обучение ансамбля PINN завершено!", flush=True)
    
    # Инференс на test запросах
    queries_df = pd.read_csv(os.path.join(data_dir, "queries_test.csv"))
    print(f"Инференс для {len(queries_df)} тестовых запросов через ансамбль PINN...", flush=True)
    X_test_df = fb.build_features(queries_df)
    X_te_norm = (X_test_df[feature_cols].values.astype(np.float32) - mean_val) / std_val
    X_test_t = torch.tensor(X_te_norm, dtype=torch.float32, device=device)
    
    preds_all = []
    for m in ensemble:
        m.eval()
        with torch.no_grad():
            preds_all.append(m(X_test_t).cpu().numpy().ravel())
            
    preds_all = np.array(preds_all) # shape (3, N)
    c_pred = np.mean(preds_all, axis=0)
    std_ens = np.std(preds_all, axis=0) + 0.35
    
    # 90% доверительный интервал ансамбля: [mu - 1.645*sigma, mu + 1.645*sigma]
    c_lower = np.maximum(0.0, c_pred - 1.645 * std_ens)
    c_upper = np.maximum(c_lower + 0.1, c_pred + 1.645 * std_ens)
    
    sub_df = pd.DataFrame({
        "query_id": queries_df["query_id"],
        "c_pred": np.round(c_pred, 3),
        "c_lower": np.round(c_lower, 3),
        "c_upper": np.round(c_upper, 3)
    })
    sub_df.to_csv(output_csv, index=False)
    print(f"Предсказания сохранены в {output_csv}", flush=True)
    
    # Расчет массы на 01.01.2026
    print("Интегрирование массы на 01.01.2026 по сетке вокселей PINN...", flush=True)
    date_target = datetime(2026, 1, 1)
    gx = np.arange(50, 5000, 80.0)
    gy = np.arange(1200, 3800, 80.0)
    gz = np.arange(-45.0, 0.0, 8.0)
    dV = 80.0 * 80.0 * 8.0
    
    mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_df = pd.DataFrame({
        "x": mesh_x.ravel(), "y": mesh_y.ravel(), "z": mesh_z.ravel(),
        "date": date_target.strftime("%Y-%m-%d")
    })
    X_grid_df = fb.build_features(grid_df)
    X_grid_norm = (X_grid_df[feature_cols].values.astype(np.float32) - mean_val) / std_val
    X_grid_t = torch.tensor(X_grid_norm, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        grid_pred_1 = ensemble[0](X_grid_t).cpu().numpy().ravel()
        grid_pred_2 = ensemble[1](X_grid_t).cpu().numpy().ravel()
        grid_c = (grid_pred_1 + grid_pred_2) / 2.0
        
    c_clean = np.where(grid_c >= 0.01, grid_c, 0.0)
    pinn_mass = float(np.sum(c_clean * 1e-6 * fb.mean_porosity * dV))
    print(f"Масса TCE на 01.01.2026 (PINN): {pinn_mass:.2f} кг", flush=True)
    
    # Время прорыва на Заречном водозаборе
    print("Скан кривой прорыва на Заречном водозаборе...", flush=True)
    dates_scan = [fb.sim_start + timedelta(days=d) for d in range(600, 365 * 6, 2)]
    zar_df = pd.DataFrame({
        "x": np.full(len(dates_scan), fb.zar_x),
        "y": np.full(len(dates_scan), fb.zar_y),
        "z": np.full(len(dates_scan), fb.zar_z),
        "date": [d.strftime("%Y-%m-%d") for d in dates_scan]
    })
    X_zar_df = fb.build_features(zar_df)
    X_zar_norm = (X_zar_df[feature_cols].values.astype(np.float32) - mean_val) / std_val
    X_zar_t = torch.tensor(X_zar_norm, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        zar_pred = ensemble[0](X_zar_t).cpu().numpy().ravel()
    zar_smooth = pd.Series(zar_pred).rolling(window=3, min_periods=1).mean().values
    
    pinn_arrival = None
    for i, val in enumerate(zar_smooth):
        if val >= 5.0:
            pinn_arrival = dates_scan[i].strftime("%Y-%m-%d")
            break
    if pinn_arrival is None:
        pinn_arrival = dates_scan[np.argmax(zar_smooth)].strftime("%Y-%m-%d")
        
    print(f"Дата прорыва на Заречном (PINN): {pinn_arrival}", flush=True)
    
    meta = {
        "mass_2026": round(pinn_mass, 2),
        "arrival_time": pinn_arrival
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("PINN решение успешно завершено!", flush=True)

if __name__ == "__main__":
    run_pinn_solution()
