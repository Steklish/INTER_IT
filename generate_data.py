"""
Скрипт генерации синтетического датасета для хакатона «Загрязнение — распространение TCE».
Моделирует 3D гидродинамический перенос загрязнителя (адвекция + дисперсия + откачка)
в водоносном горизонте с неоднородной проницаемостью и рассчитывает истинные значения
для scorer.py.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image

def generate_dataset(output_dir="data", random_seed=42):
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Геометрия расчетной области и физические константы
    # -------------------------------------------------------------
    LX, LY, LZ = 5000.0, 5000.0, 50.0   # Размеры: X (0..5000 м), Y (0..5000 м), Z (-50..0 м)
    sim_start = datetime(2021, 1, 1)
    target_mass_date = datetime(2026, 1, 1)
    
    # Источник утечки (прорванная емкость с TCE)
    source_x, source_y, source_z = 1000.0, 2500.0, -15.0
    leak_rate_kg_day = 0.25 # поступление массы в день
    
    # «Заречный водозабор»
    zarechny_x, zarechny_y, zarechny_z = 4200.0, 2700.0, -25.0
    zarechny_rate = 3500.0 # м3/сут
    
    # -------------------------------------------------------------
    # 2. Карта проницаемости (permeability.tif)
    # -------------------------------------------------------------
    # Сетка 250x250 ячеек (по 20 метров каждая)
    nx_grid, ny_grid = 250, 250
    x_coords = np.linspace(0, LX, nx_grid)
    y_coords = np.linspace(0, LY, ny_grid)
    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
    
    # Фоновая проводимость: 12 м/сут (песок)
    # Палеорусло (высокая проводимость ~35 м/сут), идущее от источника к водозабору
    # Глинистые включения (~2 м/сут) на периферии
    channel_center_y = 2500.0 + 200.0 * np.sin(X_mesh / 800.0)
    dist_to_channel = np.abs(Y_mesh - channel_center_y)
    
    K_field = 12.0 + 25.0 * np.exp(-(dist_to_channel ** 2) / (2 * 450.0 ** 2))
    # Добавим гладкий пространственный шум
    noise = np.sin(X_mesh / 500.0) * np.cos(Y_mesh / 600.0) * 3.0
    K_field = np.clip(K_field + noise, 2.0, 45.0).astype(np.float32)
    
    # Сохраняем как TIFF (Pillow)
    im = Image.fromarray(K_field)
    im.save(os.path.join(output_dir, "permeability.tif"))
    
    # Метаданные растра
    raster_meta = {
        "x_min": 0.0, "x_max": LX, "y_min": 0.0, "y_max": LY,
        "dx": LX / nx_grid, "dy": LY / ny_grid,
        "width": nx_grid, "height": ny_grid
    }
    with open(os.path.join(output_dir, "raster_meta.json"), "w", encoding="utf-8") as f:
        json.dump(raster_meta, f, indent=2)
        
    # -------------------------------------------------------------
    # 3. Источники (sources.csv) и Водозабор (abstraction.csv)
    # -------------------------------------------------------------
    sources_df = pd.DataFrame([{
        "source_id": "SRC_01",
        "x": source_x,
        "y": source_y,
        "z": source_z,
        "start_date": sim_start.strftime("%Y-%m-%d"),
        "leak_rate_kg_day": leak_rate_kg_day
    }])
    sources_df.to_csv(os.path.join(output_dir, "sources.csv"), index=False)
    
    dates_sim = [sim_start + timedelta(days=d) for d in range(0, 365 * 6 + 1, 30)]
    abstraction_records = []
    for d in dates_sim:
        # Сезонные колебания откачки на Заречном (+- 15%)
        day_of_year = d.timetuple().tm_yday
        seasonal_mult = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_year / 365.0)
        abstraction_records.append({
            "intake_name": "Zarechny_Intake",
            "x": zarechny_x,
            "y": zarechny_y,
            "z_screen_top": zarechny_z + 10.0,
            "z_screen_bottom": zarechny_z - 10.0,
            "date": d.strftime("%Y-%m-%d"),
            "rate_m3_day": round(zarechny_rate * seasonal_mult, 1)
        })
    abstraction_df = pd.DataFrame(abstraction_records)
    abstraction_df.to_csv(os.path.join(output_dir, "abstraction.csv"), index=False)
    
    # Инфильтрация осадков (recharge.csv)
    recharge_records = []
    for d in dates_sim:
        d_yday = d.timetuple().tm_yday
        # Весеннее снеготаяние дает пик питания
        r_val = 0.3 + 0.9 * np.exp(-((d_yday - 110) ** 2) / 400.0)
        recharge_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "recharge_mm_day": round(r_val, 3)
        })
    pd.DataFrame(recharge_records).to_csv(os.path.join(output_dir, "recharge.csv"), index=False)
    
    # -------------------------------------------------------------
    # 4. Мониторинговые скважины (wells.csv) и Керны (cores.csv)
    # -------------------------------------------------------------
    num_wells = 40
    well_ids = [f"W_{i+1:02d}" for i in range(num_wells)]
    
    # 15 скважин концентрируем вдоль предполагаемого шлейфа (палеорусло)
    wx_core = np.random.uniform(900, 4300, 20)
    wy_core = 2500 + 150 * np.sin(wx_core / 800.0) + np.random.normal(0, 200, 20)
    # 20 скважин рассеиваем по полигону
    wx_rand = np.random.uniform(200, 4800, 20)
    wy_rand = np.random.uniform(500, 4500, 20)
    
    wells_x = np.concatenate([wx_core, wx_rand])
    wells_y = np.concatenate([wy_core, wy_rand])
    wells_z_top = np.random.uniform(-5.0, -10.0, num_wells)
    wells_z_bottom = wells_z_top - np.random.uniform(15.0, 25.0, num_wells)
    
    wells_df = pd.DataFrame({
        "well_id": well_ids,
        "x": np.round(wells_x, 1),
        "y": np.round(wells_y, 1),
        "z_top": np.round(wells_z_top, 1),
        "z_bottom": np.round(wells_z_bottom, 1)
    })
    wells_df.to_csv(os.path.join(output_dir, "wells.csv"), index=False)
    
    # Дренажные насосы на 3 промышленных скважинах (pumping.csv)
    pump_wells = ["W_01", "W_02", "W_03"]
    pumping_records = []
    for pw in pump_wells:
        for d in dates_sim:
            pumping_records.append({
                "well_id": pw,
                "date": d.strftime("%Y-%m-%d"),
                "rate_m3_day": round(np.random.uniform(80.0, 220.0), 1)
            })
    pd.DataFrame(pumping_records).to_csv(os.path.join(output_dir, "pumping.csv"), index=False)
    
    # Керны (cores.csv)
    cores_records = []
    for wid in well_ids:
        for depth in [-10.0, -20.0, -30.0, -40.0]:
            # Эффективная пористость обычно 0.22 - 0.32
            porosity = float(np.clip(0.26 + np.random.normal(0, 0.03), 0.18, 0.35))
            lith = "sand" if porosity > 0.24 else "silt"
            cores_records.append({
                "well_id": wid,
                "depth": depth,
                "porosity": round(porosity, 3),
                "lithology": lith
            })
    pd.DataFrame(cores_records).to_csv(os.path.join(output_dir, "cores.csv"), index=False)
    
    # -------------------------------------------------------------
    # 5. Физическая функция гидродинамики и концентрации
    # -------------------------------------------------------------
    # Базовый региональный напор: уклон i_x = -0.003 (поток на восток к x=5000)
    # Напор h(x,y) = h0 - 0.003 * x + влияние депрессии от водозабора
    def compute_head(x, y, date_obj):
        days_from_start = (date_obj - sim_start).days
        seasonal_head = 1.2 * np.cos(2 * np.pi * days_from_start / 365.0)
        h_regional = 120.0 - 0.0032 * x - 0.0003 * (y - 2500.0) + seasonal_head
        # Депрессионная воронка на Заречном: s(r) = Q / (2 pi T) * ln(R/r)
        r_zar = np.maximum(np.sqrt((x - zarechny_x)**2 + (y - zarechny_y)**2), 10.0)
        drawdown = 2.5 * np.exp(-r_zar / 1200.0)
        return h_regional - drawdown

    # Физическая концентрация C(x, y, z, t):
    # Аналитическое 3D адвективно-дисперсионное решение (Гауссов пучок / шлейф):
    # Скорость переноса v_eff ~ K * i / theta ~ 25 * 0.0032 / 0.25 ~ 0.32 м/сут (~115 м/год)
    # В палеорусле скорость до 1.8-2.2 м/сут (~700 м/год), что позволяет шлейфу достигать 4200 м за 4-5 лет!
    def compute_true_concentration(x, y, z, date_obj):
        t_days = (date_obj - sim_start).days
        if t_days <= 10:
            return 0.0
            
        # Эффективная средняя скорость потока вдоль оси X
        v_flow = 1.95 # м/сут в главном русле
        # Центр пятна в момент времени t
        x_center = source_x + v_flow * t_days
        y_center = 2500.0 + 150.0 * np.sin(x_center / 800.0)
        z_center = source_z - 0.002 * (x_center - source_x) # легкое погружение тяжелого TCE
        
        # Дисперсия растет со временем
        sigma_x = np.maximum(80.0, np.sqrt(2.0 * 25.0 * t_days) + 0.04 * t_days)
        sigma_y = np.maximum(40.0, np.sqrt(2.0 * 8.0 * t_days) + 0.02 * t_days)
        sigma_z = np.maximum(6.0, np.sqrt(2.0 * 0.8 * t_days) + 0.005 * t_days)
        
        # Гауссов импульс с учетом непрерывного источника
        # Если x далеко впереди центра шлейфа, концентрация падает экспоненциально
        dx = x - x_center
        dy = y - y_center
        dz = z - z_center
        
        # Хвост источника (так как утечка постоянна, за центром тянется шлейф)
        if x < x_center and x >= source_x - 100:
            # Для точек позади фронта шлейф установившийся
            dist_trail_y = y - (2500.0 + 150.0 * np.sin(x / 800.0))
            dist_trail_z = z - (source_z - 0.002 * (x - source_x))
            sig_y_trail = 30.0 + 0.03 * (x - source_x + 100)
            sig_z_trail = 5.0 + 0.006 * (x - source_x + 100)
            c_val = 180.0 * np.exp(-0.5 * (dist_trail_y / sig_y_trail)**2 - 0.5 * (dist_trail_z / sig_z_trail)**2)
            c_val *= np.exp(-0.00035 * (x - source_x)) # постепенное разбавление
        else:
            # На переднем фронте
            exponent = -0.5 * ((dx / sigma_x)**2 + (dy / sigma_y)**2 + (dz / sigma_z)**2)
            if exponent < -25.0:
                return 0.0
            peak_c = 180.0 * (1500.0 / (1500.0 + t_days))**0.5
            c_val = peak_c * np.exp(exponent)
            
        return float(np.maximum(0.0, c_val))

    # -------------------------------------------------------------
    # 6. Замеры напора (head.csv) и концентраций (concentration.csv)
    # -------------------------------------------------------------
    # Исторический период обучения: 2021-01-01 по 2024-12-31 (4 года)
    train_dates = [sim_start + timedelta(days=d) for d in range(60, 365 * 4, 60)] # раз в 2 месяца
    
    head_records = []
    conc_records = []
    
    for _, w in wells_df.iterrows():
        wid = w["well_id"]
        wx, wy, z_t, z_b = w["x"], w["y"], w["z_top"], w["z_bottom"]
        for td in train_dates:
            # Замер напора (с добавлением инструментального шума ~2 см)
            h_true = compute_head(wx, wy, td)
            h_meas = round(h_true + np.random.normal(0, 0.03), 2)
            head_records.append({
                "well_id": wid,
                "date": td.strftime("%Y-%m-%d"),
                "head": h_meas
            })
            
            # Замеры концентраций на 2-3 интервалах фильтра
            sample_depths = np.linspace(z_t, z_b, 3)
            for z_sample in sample_depths:
                c_true = compute_true_concentration(wx, wy, z_sample, td)
                # Лабораторный шум ~5% + порог чувствительности 0.05 мкг/л
                c_noise = c_true * (1.0 + np.random.normal(0, 0.05))
                if c_noise < 0.05:
                    c_noise = 0.0
                conc_records.append({
                    "well_id": wid,
                    "z": round(z_sample, 1),
                    "date": td.strftime("%Y-%m-%d"),
                    "concentration": round(float(c_noise), 3)
                })
                
    pd.DataFrame(head_records).to_csv(os.path.join(output_dir, "head.csv"), index=False)
    pd.DataFrame(conc_records).to_csv(os.path.join(output_dir, "concentration.csv"), index=False)
    
    # -------------------------------------------------------------
    # 7. Генерация тестовых запросов (queries_test.csv) на 2025 год
    # -------------------------------------------------------------
    num_queries = 600
    test_dates = [datetime(2025, 1, 1) + timedelta(days=int(d)) for d in np.linspace(15, 350, 24)]
    
    query_x = np.random.uniform(500, 4700, num_queries)
    query_y = np.random.uniform(1500, 3500, num_queries)
    query_z = np.random.uniform(-40.0, -5.0, num_queries)
    query_d = [np.random.choice(test_dates) for _ in range(num_queries)]
    
    queries_df = pd.DataFrame({
        "query_id": np.arange(1, num_queries + 1),
        "x": np.round(query_x, 1),
        "y": np.round(query_y, 1),
        "z": np.round(query_z, 1),
        "date": [d.strftime("%Y-%m-%d") for d in query_d]
    })
    queries_df.to_csv(os.path.join(output_dir, "queries_test.csv"), index=False)
    
    # Истинные значения для проверки жюри (ground_truth_test.csv)
    c_ground_truth = []
    for _, q in queries_df.iterrows():
        c_gt = compute_true_concentration(q["x"], q["y"], q["z"], datetime.strptime(q["date"], "%Y-%m-%d"))
        c_ground_truth.append(round(c_gt, 3))
        
    gt_df = pd.DataFrame({
        "query_id": queries_df["query_id"],
        "c_true": c_ground_truth
    })
    gt_df.to_csv(os.path.join(output_dir, "ground_truth_test.csv"), index=False)
    
    # -------------------------------------------------------------
    # 8. Расчет истинных целевых ответов жюри: Масса на 2026 и Время прорыва
    # -------------------------------------------------------------
    print("Вычисление ground truth массы на 01.01.2026...")
    # Интегрирование по равномерной 3D сетке:
    # Шаг dx=50 м, dy=50 м, dz=2.5 м -> dV = 50 * 50 * 2.5 = 6250 м3
    grid_gx = np.arange(25, 5000, 50)
    grid_gy = np.arange(25, 5000, 50)
    grid_gz = np.arange(-47.5, 0, 5.0)
    dV = 50.0 * 50.0 * 5.0 # м3
    mean_porosity = 0.26 # средняя пористость
    
    total_mass_kg = 0.0
    for gz in grid_gz:
        for gy in grid_gy:
            # Оптимизация: берем только полосу возле русла
            if abs(gy - 2500) > 1200:
                continue
            for gx in grid_gx:
                c_val = compute_true_concentration(gx, gy, gz, target_mass_date)
                if c_val > 0.01:
                    # c_val в мкг/л = 1e-6 г/л = 1e-3 г/м3 = 1e-6 кг/м3
                    mass_cell_kg = (c_val * 1e-6) * (mean_porosity * dV)
                    total_mass_kg += mass_cell_kg
                    
    true_mass_2026 = round(float(total_mass_kg), 2)
    print(f"Ground Truth масса TCE на 01.01.2026: {true_mass_2026:.2f} кг")
    
    # Определение точного момента прорыва на Заречном водозаборе (порог 5.0 мкг/л)
    print("Вычисление ground truth даты прорыва порога 5 мкг/л...")
    arrival_date_str = None
    arrival_days = None
    for d in range(100, 365 * 8): # проверяем каждый день до 2029 года
        cur_date = sim_start + timedelta(days=d)
        c_zar = compute_true_concentration(zarechny_x, zarechny_y, zarechny_z, cur_date)
        if c_zar >= 5.0:
            arrival_date_str = cur_date.strftime("%Y-%m-%d")
            arrival_days = d
            break
            
    print(f"Ground Truth дата прорыва на Заречном: {arrival_date_str} (день {arrival_days})")
    
    gt_meta = {
        "mass_2026_kg": true_mass_2026,
        "arrival_time_date": arrival_date_str,
        "arrival_time_days_from_start": arrival_days,
        "threshold_ug_l": 5.0,
        "sim_start": sim_start.strftime("%Y-%m-%d"),
        "zarechny_intake": {"x": zarechny_x, "y": zarechny_y, "z": zarechny_z}
    }
    with open(os.path.join(output_dir, "ground_truth_meta.json"), "w", encoding="utf-8") as f:
        json.dump(gt_meta, f, indent=2)
        
    print(f"Генерация датасета завершена успешно в папку '{output_dir}'.")

if __name__ == "__main__":
    generate_dataset()
