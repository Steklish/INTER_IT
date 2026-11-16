"""
Скрипт оценки качества решений жюри (scorer.py).
Оценивает 4 выхода модели:
1. c_pred: Точечный прогноз (RMSE)
2. [c_lower, c_upper]: Калибровка 90%-го доверительного интервала (Coverage + Interval Score)
3. mass_2026: Оценка массы загрязнителя на 01.01.2026 (Относительная ошибка)
4. arrival_time: Дата достижения 5 мкг/л на Заречном водозаборе (Ошибка в днях)
Итоговый взвешенный балл: Score из 100.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

def evaluate(submission_csv_path, submission_meta_path, data_dir="data"):
    # Проверка наличия файлов
    gt_test_path = os.path.join(data_dir, "ground_truth_test.csv")
    gt_meta_path = os.path.join(data_dir, "ground_truth_meta.json")
    
    if not os.path.exists(submission_csv_path):
        raise FileNotFoundError(f"Файл с предсказаниями не найден: {submission_csv_path}")
    if not os.path.exists(submission_meta_path):
        raise FileNotFoundError(f"Файл с мета-ответами не найден: {submission_meta_path}")
    if not os.path.exists(gt_test_path) or not os.path.exists(gt_meta_path):
        raise FileNotFoundError(f"Файлы ground truth не найдены в {data_dir}")
        
    sub_df = pd.read_csv(submission_csv_path)
    gt_df = pd.read_csv(gt_test_path)
    
    with open(submission_meta_path, "r", encoding="utf-8") as f:
        sub_meta = json.load(f)
    with open(gt_meta_path, "r", encoding="utf-8") as f:
        gt_meta = json.load(f)
        
    # Сверка query_id
    merged = pd.merge(gt_df, sub_df, on="query_id", suffixes=("_true", "_pred"))
    if len(merged) != len(gt_df):
        raise ValueError(f"Размерность ответов ({len(merged)}) не совпадает с тестом ({len(gt_df)})!")
        
    c_true = merged["c_true"].values
    c_pred = merged["c_pred"].values
    c_lower = merged["c_lower"].values
    c_upper = merged["c_upper"].values
    
    # -------------------------------------------------------------
    # 1. Точечный прогноз (RMSE)
    # -------------------------------------------------------------
    rmse = float(np.sqrt(np.mean((c_pred - c_true) ** 2)))
    sigma_true = float(np.std(c_true)) if np.std(c_true) > 0 else 1.0
    # Нормированный балл от 0 до 100
    s_rmse = float(np.clip(100.0 * np.exp(-rmse / (0.35 * sigma_true)), 0.0, 100.0))
    
    # -------------------------------------------------------------
    # 2. Калибровка 90% интервала (Coverage + Interval Score)
    # -------------------------------------------------------------
    in_interval = (c_true >= c_lower) & (c_true <= c_upper)
    coverage = float(np.mean(in_interval))
    
    # Winkler Score для 90% интервала (alpha = 0.10)
    alpha = 0.10
    widths = c_upper - c_lower
    # Проверка корректности границ: нижняя не должна быть выше верхней
    invalid_bounds = np.sum(widths < 0)
    if invalid_bounds > 0:
        penalty_invalid = 0.5
    else:
        penalty_invalid = 1.0
        
    penalty_under = (2.0 / alpha) * np.maximum(0.0, c_lower - c_true)
    penalty_over = (2.0 / alpha) * np.maximum(0.0, c_true - c_upper)
    winkler_scores = widths + penalty_under + penalty_over
    mean_winkler = float(np.mean(winkler_scores))
    
    # Штраф за отклонение от покрытия 90%
    coverage_penalty = float(np.exp(-20.0 * abs(coverage - 0.90)))
    # Нормализация скора интервала
    s_interval = float(np.clip(100.0 * (1.0 / (1.0 + mean_winkler / (sigma_true + 1e-5))) * coverage_penalty * penalty_invalid, 0.0, 100.0))
    
    # -------------------------------------------------------------
    # 3. Оценка массы загрязнителя на 01.01.2026
    # -------------------------------------------------------------
    true_mass = float(gt_meta["mass_2026_kg"])
    pred_mass = float(sub_meta.get("mass_2026", 0.0))
    rel_mass_err = float(abs(pred_mass - true_mass) / (true_mass + 1e-5))
    s_mass = float(np.clip(100.0 * max(0.0, 1.0 - 2.5 * rel_mass_err), 0.0, 100.0))
    
    # -------------------------------------------------------------
    # 4. Время прихода к Заречному водозабору
    # -------------------------------------------------------------
    true_arrival_str = gt_meta["arrival_time_date"]
    true_arrival_date = datetime.strptime(true_arrival_str, "%Y-%m-%d")
    
    pred_arrival_val = sub_meta.get("arrival_time")
    if isinstance(pred_arrival_val, (int, float)):
        # Если передано число дней от sim_start
        sim_start_date = datetime.strptime(gt_meta["sim_start"], "%Y-%m-%d")
        pred_arrival_date = sim_start_date + timedelta(days=float(pred_arrival_val))
    elif isinstance(pred_arrival_val, str):
        try:
            pred_arrival_date = datetime.strptime(pred_arrival_val.strip(), "%Y-%m-%d")
        except Exception:
            pred_arrival_date = datetime(2099, 1, 1)
    else:
        pred_arrival_date = datetime(2099, 1, 1)
        
    days_diff = abs((pred_arrival_date - true_arrival_date).days)
    s_arrival = float(np.clip(100.0 * np.exp(-days_diff / 45.0), 0.0, 100.0))
    
    # -------------------------------------------------------------
    # Финальный композитный скор (из 100)
    # -------------------------------------------------------------
    total_score = (
        0.35 * s_rmse + 
        0.25 * s_interval + 
        0.20 * s_mass + 
        0.20 * s_arrival
    )
    
    metrics = {
        "total_score": round(total_score, 2),
        "point_forecast": {
            "rmse": round(rmse, 4),
            "sigma_true": round(sigma_true, 4),
            "score": round(s_rmse, 2)
        },
        "interval_calibration": {
            "coverage": round(coverage * 100, 2),
            "target_coverage": 90.0,
            "mean_interval_width": round(float(np.mean(widths)), 3),
            "mean_winkler_loss": round(mean_winkler, 3),
            "score": round(s_interval, 2)
        },
        "mass_estimation": {
            "pred_mass_kg": round(pred_mass, 2),
            "true_mass_kg": round(true_mass, 2),
            "rel_error_pct": round(rel_mass_err * 100, 2),
            "score": round(s_mass, 2)
        },
        "arrival_time": {
            "pred_arrival": pred_arrival_date.strftime("%Y-%m-%d"),
            "true_arrival": true_arrival_str,
            "diff_days": days_diff,
            "score": round(s_arrival, 2)
        }
    }
    
    return metrics

def print_report(metrics):
    print("\n" + "="*60)
    print("           ИТОГОВЫЙ ОТЧЕТ ЖЮРИ (SCORER.PY)           ")
    print("="*60)
    print(f"★ ФИНАЛЬНЫЙ БАЛЛ РЕШЕНИЯ: {metrics['total_score']} / 100.00\n")
    
    m_pt = metrics["point_forecast"]
    print(f"1. Точечный прогноз (RMSE)        [Вес: 35%]:")
    print(f"   - RMSE: {m_pt['rmse']} мкг/л (при std={m_pt['sigma_true']})")
    print(f"   - Балл: {m_pt['score']} / 100")
    
    m_int = metrics["interval_calibration"]
    print(f"\n2. Доверительный интервал (90%)     [Вес: 25%]:")
    print(f"   - Эмпирическое покрытие: {m_int['coverage']}% (Цель: {m_int['target_coverage']}%)")
    print(f"   - Средняя ширина интервала: {m_int['mean_interval_width']} мкг/л")
    print(f"   - Балл: {m_int['score']} / 100")
    
    m_m = metrics["mass_estimation"]
    print(f"\n3. Оценка массы на 01.01.2026       [Вес: 20%]:")
    print(f"   - Предсказано: {m_m['pred_mass_kg']} кг (Истина: {m_m['true_mass_kg']} кг)")
    print(f"   - Относительная ошибка: {m_m['rel_error_pct']}%")
    print(f"   - Балл: {m_m['score']} / 100")
    
    m_arr = metrics["arrival_time"]
    print(f"\n4. Дата прорыва на Заречном         [Вес: 20%]:")
    print(f"   - Предсказана дата: {m_arr['pred_arrival']} (Истина: {m_arr['true_arrival']})")
    print(f"   - Ошибка во времени: {m_arr['diff_days']} дн.")
    print(f"   - Балл: {m_arr['score']} / 100")
    print("="*60 + "\n")

if __name__ == "__main__":
    sub_csv = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    sub_meta = sys.argv[2] if len(sys.argv) > 2 else "submission_meta.json"
    data_p = sys.argv[3] if len(sys.argv) > 3 else "data"
    
    res = evaluate(sub_csv, sub_meta, data_dir=data_p)
    print_report(res)
