"""
Сводный бенчмарк всех 5 алгоритмов:
1. Baseline (Эвристика)
2. GPR / Kriging (Геостатистика и Гауссовские процессы)
3. LightGBM + Physics + CQR (Табличный градиентный бустинг)
4. Numerical Simulator (Численный гидродинамический симулятор)
5. PINN (Физико-информированные нейросети на PyTorch / CUDA)
"""

import os
import json
import pandas as pd
from scorer import evaluate

def run_full_benchmark():
    print("=" * 85)
    print("      ИТОГОВАЯ БИТВА АЛГОРИТМОВ: СРАВНЕНИЕ ВСЕХ 5 РЕШЕНИЙ ДЛЯ ТАСКИ TCE      ")
    print("=" * 85)
    
    solutions = [
        ("1. Baseline", "submission_baseline.csv", "submission_baseline_meta.json"),
        ("2. GPR / Kriging", "submission_gpr.csv", "submission_gpr_meta.json"),
        ("3. LightGBM + CQR", "submission.csv", "submission_meta.json"),
        ("4. Numerical Simulator", "submission_numsim.csv", "submission_numsim_meta.json"),
        ("5. PINN (PyTorch GPU)", "submission_pinn.csv", "submission_pinn_meta.json"),
    ]
    
    rows = []
    for name, csv_p, meta_p in solutions:
        if not os.path.exists(csv_p) or not os.path.exists(meta_p):
            print(f"Пропуск {name}: файлы {csv_p} не найдены.")
            continue
            
        m = evaluate(csv_p, meta_p, "data")
        rows.append({
            "Алгоритм": name,
            "Итоговый Score": f"{m['total_score']} / 100",
            "RMSE (мкг/л)": f"{m['point_forecast']['rmse']}",
            "Покрытие 90%": f"{m['interval_calibration']['coverage']}%",
            "Ширина интервала": f"{m['interval_calibration']['mean_interval_width']} мкг/л",
            "Ошибка массы (%)": f"{m['mass_estimation']['rel_error_pct']}%",
            "Ошибка даты (дни)": f"{m['arrival_time']['diff_days']} дн."
        })
        
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print("=" * 85 + "\n")
    
    # Сохраняем в JSON
    df.to_json("benchmark_summary.json", orient="records", indent=2)

if __name__ == "__main__":
    run_full_benchmark()
