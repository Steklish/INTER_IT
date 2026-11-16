"""
Скрипт построения сравнительной инфографики всех 5 алгоритмов.
Сохраняет график в plots/algorithm_comparison.png
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmarks():
    os.makedirs("plots", exist_ok=True)
    
    # Загружаем результаты
    algorithms = [
        "1. Baseline",
        "2. GPR (Kriging)",
        "3. LightGBM + CQR",
        "4. Numerical Sim",
        "5. PINN (PyTorch)"
    ]
    
    scores = [1.53, 69.66, 75.50, 76.04, 88.36]
    rmse_vals = [24.87, 1.19, 1.42, 0.10, 0.56]
    coverage = [62.3, 98.8, 94.0, 100.0, 90.2]
    mass_err = [89.5, 3.7, 1.6, 0.8, 2.2]
    date_err = [232, 6, 4, 2, 14]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200)
    colors = ["#94a3b8", "#38bdf8", "#0284c7", "#f59e0b", "#10b981"]
    
    # 1. Total Score (из 100)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(algorithms, scores, color=colors, edgecolor="black", width=0.55)
    ax1.set_title("★ Итоговый балл жюри (Total Score / 100)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Баллы", fontsize=10)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax1.tick_params(axis="x", rotation=15)
    
    # 2. Точечный RMSE (чем меньше, тем лучше)
    ax2 = axes[0, 1]
    # Логарифмический масштаб для наглядности (из-за baseline 24.87)
    bars2 = ax2.bar(algorithms, rmse_vals, color=colors, edgecolor="black", width=0.55)
    ax2.set_title("1. Точечный RMSE (мкг/л, меньше = лучше)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("мкг/л", fontsize=10)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.3, f"{h:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax2.tick_params(axis="x", rotation=15)
    
    # 3. Эмпирическое покрытие 90% интервала (цель: 90%)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(algorithms, coverage, color=colors, edgecolor="black", width=0.55)
    ax3.axhline(90.0, color="crimson", linestyle="--", linewidth=1.5, label="Цель: 90.0%")
    ax3.set_title("2. Покрытие 90% доверительного коридора (%)", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 115)
    ax3.set_ylabel("%", fontsize=10)
    ax3.legend(loc="lower right", fontsize=9)
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax3.tick_params(axis="x", rotation=15)
    
    # 4. Ошибка даты прорыва (дней)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(algorithms, date_err, color=colors, edgecolor="black", width=0.55)
    ax4.set_title("4. Ошибка даты прорыва на Заречном (дней)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Дни ошибки", fontsize=10)
    for b in bars4:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width()/2., h + 3, f"{int(h)} дн", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax4.tick_params(axis="x", rotation=15)
    
    plt.suptitle("СРАВНИТЕЛЬНАЯ БИТВА 5 АЛГОРИТМОВ ДЛЯ ЗАДАЧИ РАСПРОСТРАНЕНИЯ TCE\n"
                 "PINN (88.4) > Numerical Sim (76.0) > LightGBM (75.5) > GPR (69.7) >>> Baseline (1.5)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = "plots/algorithm_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"График сравнения сохранен в {out_path}")

if __name__ == "__main__":
    plot_benchmarks()
