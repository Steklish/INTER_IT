"""
Базовое решение (baseline.py).
Простой эвристический подход:
1. c_pred: Медианное значение замеров по ближайшей скважине.
2. [c_lower, c_upper]: Фиксированный разброс вокруг прогноза.
3. mass_2026: Грубая эвристическая оценка объема шлейфа.
4. arrival_time: Линейная экстраполяция до водозабора.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

def run_baseline(data_dir="data", output_csv="submission_baseline.csv", output_meta="submission_baseline_meta.json"):
    print("Запуск Baseline решения...")
    
    # 1. Загрузка данных
    wells_df = pd.read_csv(os.path.join(data_dir, "wells.csv"))
    conc_df = pd.read_csv(os.path.join(data_dir, "concentration.csv"))
    queries_df = pd.read_csv(os.path.join(data_dir, "queries_test.csv"))
    
    # Расчет среднего значения концентрации для каждой скважины
    well_mean_conc = conc_df.groupby("well_id")["concentration"].mean().to_dict()
    global_mean = conc_df["concentration"].mean()
    global_std = conc_df["concentration"].std()
    
    # Сопоставление скважины к координатам
    well_coords = wells_df[["well_id", "x", "y"]].set_index("well_id")
    
    c_pred_list = []
    c_lower_list = []
    c_upper_list = []
    
    # 2. Для каждого запроса находим ближайшую скважину
    for _, q in queries_df.iterrows():
        qx, qy = q["x"], q["y"]
        dists = np.sqrt((well_coords["x"] - qx)**2 + (well_coords["y"] - qy)**2)
        nearest_well = dists.idxmin()
        min_dist = dists.min()
        
        # Затухание с расстоянием
        decay = np.exp(-min_dist / 800.0)
        base_val = well_mean_conc.get(nearest_well, global_mean)
        pred_val = max(0.0, float(base_val * decay))
        
        # Наивный коридор (фиксированная неопределенность)
        margin = max(1.5, pred_val * 0.6)
        c_pred_list.append(round(pred_val, 3))
        c_lower_list.append(round(max(0.0, pred_val - margin), 3))
        c_upper_list.append(round(pred_val + margin, 3))
        
    sub_df = pd.DataFrame({
        "query_id": queries_df["query_id"],
        "c_pred": c_pred_list,
        "c_lower": c_lower_list,
        "c_upper": c_upper_list
    })
    sub_df.to_csv(output_csv, index=False)
    print(f"Предсказания baseline сохранены в {output_csv}")
    
    # 3. Наивная оценка массы и даты прихода
    # Масса: грубая оценка среднего объема
    naive_mass = round(float(global_mean * 0.25 * 5000 * 5000 * 30 * 1e-6 * 0.05), 2)
    # Дата: предположим середину 2025 года
    naive_arrival = "2025-06-01"
    
    meta = {
        "mass_2026": naive_mass,
        "arrival_time": naive_arrival
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Мета-данные baseline сохранены в {output_meta}")

if __name__ == "__main__":
    run_baseline()
