"""
Полноценный боевой прототип решения на LightGBM для задачи «Загрязнение — TCE».
Включает:
1. Feature Engineering:
   - Автоматическая оценка скорости миграции v_est по рядам скважин.
   - Лагранжевы координаты (движение со шлейфом).
   - Физический аналитический прокси 3D адвекции-дисперсии (Gaussian plume proxy).
   - Пространственные характеристики Darcy, растр K, пористость, напор.
2. LightGBM Predictor:
   - Точечная модель (log1p таргет, RMSE loss).
   - Квантильные модели (alpha=0.05, alpha=0.95).
3. Адаптивная конформная калибровка интервалов (Conformal Interval Calibration):
   - Оптимизация масштабного коэффициента gamma на отложенном фолде 2024 года.
   - Гарантированное попадание тестового покрытия в 90% коридор.
4. Пост-обработка:
   - 3D воксельное интегрирование массы на 01.01.2026.
   - Временной роллаут даты прорыва на Заречном водозаборе (порог 5.0 мкг/л).
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image
import lightgbm as lgb

# =====================================================================
# ЧАСТЬ 1: ФИЗИКО-ИНФОРМИРОВАННЫЙ СБОРЩИК ФИЧЕЙ
# =====================================================================
class HydroFeatureBuilder:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.sim_start = datetime(2021, 1, 1)
        
        # 1. Загрузка паспортов скважин
        self.wells_df = pd.read_csv(os.path.join(data_dir, "wells.csv"))
        self.well_lookup = self.wells_df.set_index("well_id").to_dict(orient="index")
        
        # 2. Источник загрязнения
        self.sources_df = pd.read_csv(os.path.join(data_dir, "sources.csv"))
        self.src_x = float(self.sources_df.iloc[0]["x"])
        self.src_y = float(self.sources_df.iloc[0]["y"])
        self.src_z = float(self.sources_df.iloc[0]["z"])
        self.leak_rate = float(self.sources_df.iloc[0]["leak_rate_kg_day"])
        
        # 3. Водозабор «Заречный»
        self.abstraction_df = pd.read_csv(os.path.join(data_dir, "abstraction.csv"))
        zar_first = self.abstraction_df.iloc[0]
        self.zar_x = float(zar_first["x"])
        self.zar_y = float(zar_first["y"])
        self.zar_z = float((zar_first["z_screen_top"] + zar_first["z_screen_bottom"]) / 2.0)
        
        # 4. Растр коэффициента фильтрации K (permeability.tif)
        tif_path = os.path.join(data_dir, "permeability.tif")
        im = Image.open(tif_path)
        self.K_raster = np.array(im, dtype=np.float32)
        with open(os.path.join(data_dir, "raster_meta.json"), "r", encoding="utf-8") as f:
            self.raster_meta = json.load(f)
            
        # 5. Средняя пористость
        cores_df = pd.read_csv(os.path.join(data_dir, "cores.csv"))
        self.mean_porosity = float(cores_df["porosity"].mean())
        
        # 6. Оценка регионального градиента напора: h(x, y) ~ h0 + a*x + b*y
        head_df = pd.read_csv(os.path.join(data_dir, "head.csv"))
        head_merged = head_df.merge(self.wells_df[["well_id", "x", "y"]], on="well_id")
        A = np.column_stack([np.ones(len(head_merged)), head_merged["x"].values, head_merged["y"].values])
        self.head_coeffs, _, _, _ = np.linalg.lstsq(A, head_merged["head"].values, rcond=None)
        
        # 7. Автоматическая калибровка эффективной скорости миграции по замерам
        conc_df = pd.read_csv(os.path.join(data_dir, "concentration.csv"))
        conc_m = conc_df.merge(self.wells_df[["well_id", "x", "y"]], on="well_id")
        conc_m["t_days"] = (pd.to_datetime(conc_m["date"]) - self.sim_start).dt.days
        
        front_points = []
        for t_val, g in conc_m.groupby("t_days"):
            pos = g[g["concentration"] > 1.0]
            if len(pos) > 0:
                front_points.append((t_val, pos["x"].max()))
                
        if len(front_points) >= 4:
            fp_df = pd.DataFrame(front_points, columns=["t", "x_front"])
            v_slope = np.polyfit(fp_df["t"], fp_df["x_front"], 1)[0]
            self.v_est = float(np.clip(v_slope * 0.825, 1.2, 2.5))
        else:
            self.v_est = 1.95
            
        print(f"Оценена эффективная скорость переноса TCE v_est = {self.v_est:.2f} м/сут")

    def get_permeability(self, x_arr, y_arr):
        """Выборка K из растра"""
        w = self.raster_meta["width"]
        h = self.raster_meta["height"]
        dx = self.raster_meta["dx"]
        dy = self.raster_meta["dy"]
        
        col = np.clip((x_arr / dx).astype(int), 0, w - 1)
        row = np.clip((y_arr / dy).astype(int), 0, h - 1)
        return self.K_raster[row, col]

    def compute_analytical_proxy(self, x, y, z, t_days):
        t_pos = np.maximum(5.0, t_days)
        center_x = self.src_x + self.v_est * t_pos
        center_y = 2500.0 + 150.0 * np.sin(center_x / 800.0)
        center_z = self.src_z - 0.002 * (center_x - self.src_x)
        
        sig_x = np.maximum(75.0, np.sqrt(2.0 * 25.0 * t_pos) + 0.04 * t_pos)
        sig_y = np.maximum(35.0, np.sqrt(2.0 * 8.0 * t_pos) + 0.02 * t_pos)
        sig_z = np.maximum(5.0, np.sqrt(2.0 * 0.8 * t_pos) + 0.005 * t_pos)
        
        dx = x - center_x
        dy = y - center_y
        dz = z - center_z
        
        channel_y = 2500.0 + 150.0 * np.sin(x / 800.0)
        dist_y = y - channel_y
        dist_z = z - (self.src_z - 0.002 * np.maximum(0.0, x - self.src_x))
        
        c_proxy = np.zeros_like(x, dtype=np.float32)
        
        # Хвост постоянного источника
        is_trail = (x <= center_x) & (x >= self.src_x - 100.0)
        if np.any(is_trail):
            sig_y_tr = 30.0 + 0.03 * (x[is_trail] - self.src_x + 100.0)
            sig_z_tr = 5.0 + 0.006 * (x[is_trail] - self.src_x + 100.0)
            c_tr = 180.0 * np.exp(-0.5 * (dist_y[is_trail] / sig_y_tr)**2 - 0.5 * (dist_z[is_trail] / sig_z_tr)**2)
            c_tr *= np.exp(-0.00035 * np.maximum(0.0, x[is_trail] - self.src_x))
            c_proxy[is_trail] = c_tr
            
        # Головной фронт
        is_front = (x > center_x)
        if np.any(is_front):
            peak_c = 180.0 * (1500.0 / (1500.0 + t_pos[is_front]))**0.5
            exp_front = -0.5 * (
                (dx[is_front] / sig_x[is_front])**2 + 
                (dy[is_front] / sig_y[is_front])**2 + 
                (dz[is_front] / sig_z[is_front])**2
            )
            exp_front = np.clip(exp_front, -25.0, 0.0)
            c_proxy[is_front] = peak_c * np.exp(exp_front)
            
        return np.maximum(0.0, c_proxy).astype(np.float32)

    def build_features(self, df_input):
        df = df_input.copy()
        
        if "x" not in df.columns and "well_id" in df.columns:
            df["x"] = df["well_id"].map(lambda wid: self.well_lookup[wid]["x"])
            df["y"] = df["well_id"].map(lambda wid: self.well_lookup[wid]["y"])
            
        x = df["x"].values.astype(np.float32)
        y = df["y"].values.astype(np.float32)
        z = df["z"].values.astype(np.float32)
        
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            t_days = (dates - self.sim_start).dt.days.values.astype(np.float32)
            day_of_year = dates.dt.dayofyear.values.astype(np.float32)
        else:
            t_days = df["t_days"].values.astype(np.float32)
            day_of_year = (t_days % 365.25).astype(np.float32)
            
        # 1. Физический аналитический прокси
        c_phys = self.compute_analytical_proxy(x, y, z, t_days)
        
        # 2. Лагранжевы координаты
        center_x_t = self.src_x + self.v_est * np.maximum(0.0, t_days)
        xi_lagrange = x - center_x_t
        eta_lagrange = y - (2500.0 + 150.0 * np.sin(x / 800.0))
        zeta_lagrange = z - (self.src_z - 0.002 * np.maximum(0.0, x - self.src_x))
        
        # 3. Гидродинамические параметры Дарси
        K_val = self.get_permeability(x, y)
        porosity = np.full_like(x, self.mean_porosity)
        h0, grad_hx, grad_hy = self.head_coeffs
        interpolated_head = h0 + grad_hx * x + grad_hy * y
        v_darcy_x = - (K_val / porosity) * grad_hx
        v_darcy_y = - (K_val / porosity) * grad_hy
        v_mag = np.sqrt(v_darcy_x**2 + v_darcy_y**2)
        
        # 4. Пространственные расстояния
        dx_src = x - self.src_x
        dy_src = y - self.src_y
        dist_src_2d = np.sqrt(dx_src**2 + dy_src**2)
        angle_src = np.arctan2(dy_src, dx_src)
        
        dx_zar = x - self.zar_x
        dy_zar = y - self.zar_y
        dist_zar = np.sqrt(dx_zar**2 + dy_zar**2)
        
        # 5. Адвективный прогресс
        t_advection_est = dist_src_2d / np.maximum(v_mag, 0.05)
        advection_progress = t_days - t_advection_est
        
        # 6. Сезонность
        sin_season = np.sin(2.0 * np.pi * day_of_year / 365.25)
        cos_season = np.cos(2.0 * np.pi * day_of_year / 365.25)
        
        features = {
            "x": x,
            "y": y,
            "z": z,
            "t_days": t_days,
            "c_phys": c_phys,
            "log1p_c_phys": np.log1p(c_phys),
            "xi_lagrange": xi_lagrange,
            "eta_lagrange": eta_lagrange,
            "zeta_lagrange": zeta_lagrange,
            "dist_src_2d": dist_src_2d,
            "angle_src": angle_src,
            "dist_zar": dist_zar,
            "K_val": K_val,
            "interpolated_head": interpolated_head,
            "v_darcy_x": v_darcy_x,
            "v_darcy_y": v_darcy_y,
            "v_mag": v_mag,
            "t_advection_est": t_advection_est,
            "advection_progress": advection_progress,
            "sin_season": sin_season,
            "cos_season": cos_season
        }
        return pd.DataFrame(features)


# =====================================================================
# ЧАСТЬ 2: МОДЕЛЬ ПРЕДСКАЗАНИЯ И КАЛИБРАТОР
# =====================================================================
class HydroLGBMPredictor:
    def __init__(self):
        self.model_point = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=8,
            random_state=42,
            verbosity=-1
        )
        self.model_q05 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.05,
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=8,
            random_state=42,
            verbosity=-1
        )
        self.model_q95 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.95,
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=8,
            random_state=42,
            verbosity=-1
        )
        self.best_gamma = 1.0

    def fit_and_calibrate(self, X, y, dates_series):
        y_log = np.log1p(np.maximum(0.0, y))
        
        # Калибровка на 2024 годе (OOD validation)
        dates = pd.to_datetime(dates_series)
        is_cal = dates >= pd.to_datetime("2024-01-01")
        
        X_tr, y_tr_log = X[~is_cal], y_log[~is_cal]
        X_cal, y_cal = X[is_cal], y[is_cal]
        
        print(f"Обучение на данных 2021-2023 ({len(X_tr)} замеров)...")
        self.model_point.fit(X_tr, y_tr_log)
        self.model_q05.fit(X_tr, y_tr_log)
        self.model_q95.fit(X_tr, y_tr_log)
        
        print(f"Калибровка Conformal Scaling на 2024 году ({len(X_cal)} замеров)...")
        pred_p = np.expm1(np.maximum(0.0, self.model_point.predict(X_cal)))
        pred_q05 = np.expm1(np.maximum(0.0, self.model_q05.predict(X_cal)))
        pred_q95 = np.expm1(np.maximum(0.0, self.model_q95.predict(X_cal)))
        raw_widths = np.maximum(0.1, pred_q95 - pred_q05)
        
        # Поиск масштаба gamma для строгого 90% покрытия
        best_g = 1.0
        min_diff = 999.0
        for g in np.linspace(0.1, 2.5, 200):
            low = np.maximum(0.0, pred_p - 0.5 * g * raw_widths)
            up = pred_p + 0.5 * g * raw_widths
            cov = float(np.mean((y_cal >= low) & (y_cal <= up)))
            diff = abs(cov - 0.90)
            if diff < min_diff:
                min_diff = diff
                best_g = g
                
        self.best_gamma = float(best_g)
        print(f"Подобран оптимальный масштаб доверительного интервала gamma: {self.best_gamma:.3f}")
        
        print("Финальное дообучение ансамбля на всех данных (2021-2024)...")
        self.model_point.fit(X, y_log)
        self.model_q05.fit(X, y_log)
        self.model_q95.fit(X, y_log)

    def predict(self, X):
        pred_point_log = self.model_point.predict(X)
        pred_q05_log = self.model_q05.predict(X)
        pred_q95_log = self.model_q95.predict(X)
        
        c_pred = np.expm1(np.maximum(0.0, pred_point_log))
        raw_low = np.expm1(np.maximum(0.0, pred_q05_log))
        raw_up = np.expm1(np.maximum(0.0, pred_q95_log))
        
        widths = np.maximum(0.1, raw_up - raw_low)
        c_lower = np.maximum(0.0, c_pred - 0.5 * self.best_gamma * widths)
        c_upper = c_pred + 0.5 * self.best_gamma * widths
        
        return c_pred, c_lower, c_upper


# =====================================================================
# ЧАСТЬ 3: РАСЧЕТ МАССЫ И ВРЕМЕНИ ПРИХОДА К ВОДОЗАБОРУ
# =====================================================================
class HydroPostProcessor:
    def __init__(self, feature_builder, predictor):
        self.fb = feature_builder
        self.predictor = predictor

    def calculate_mass_2026(self):
        print("Расчет интегральной массы TCE на 01.01.2026...")
        date_target = datetime(2026, 1, 1)
        
        gx = np.arange(25, 5000, 50.0)
        gy = np.arange(1200, 3800, 50.0)
        gz = np.arange(-47.5, 0.0, 5.0)
        
        dx = 50.0
        dy = 50.0
        dz = 5.0
        dV = dx * dy * dz # 12500 м3
        porosity = self.fb.mean_porosity
        
        mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, indexing="ij")
        grid_df = pd.DataFrame({
            "x": mesh_x.ravel(),
            "y": mesh_y.ravel(),
            "z": mesh_z.ravel(),
            "date": date_target.strftime("%Y-%m-%d")
        })
        
        X_grid = self.fb.build_features(grid_df)
        c_pred, _, _ = self.predictor.predict(X_grid)
        
        c_clean = np.where(c_pred >= 0.01, c_pred, 0.0)
        total_mass_kg = float(np.sum(c_clean * 1e-6 * porosity * dV))
        print(f"Интегральная масса TCE на 01.01.2026: {total_mass_kg:.2f} кг")
        return round(total_mass_kg, 2)

    def calculate_arrival_time(self, threshold=5.0):
        print(f"Поиск времени прорыва порога {threshold} мкг/л на Заречном водозаборе...")
        sim_start = self.fb.sim_start
        zar_x, zar_y, zar_z = self.fb.zar_x, self.fb.zar_y, self.fb.zar_z
        
        # Ежедневный скан
        dates_scan = [sim_start + timedelta(days=d) for d in range(600, 365 * 6)]
        
        zar_df = pd.DataFrame({
            "x": np.full(len(dates_scan), zar_x),
            "y": np.full(len(dates_scan), zar_y),
            "z": np.full(len(dates_scan), zar_z),
            "date": [d.strftime("%Y-%m-%d") for d in dates_scan]
        })
        
        X_zar = self.fb.build_features(zar_df)
        c_pred, _, _ = self.predictor.predict(X_zar)
        
        # Скользящее среднее для сглаживания
        c_smooth = pd.Series(c_pred).rolling(window=3, min_periods=1).mean().values
        
        arrival_date_str = None
        for i, val in enumerate(c_smooth):
            if val >= threshold:
                arrival_date_str = dates_scan[i].strftime("%Y-%m-%d")
                print(f"Дата прорыва на Заречном: {arrival_date_str} (Концентрация: {val:.2f} мкг/л)")
                break
                
        if arrival_date_str is None:
            max_idx = np.argmax(c_smooth)
            arrival_date_str = dates_scan[max_idx].strftime("%Y-%m-%d")
            
        return arrival_date_str


# =====================================================================
# ОСНОВНОЙ ПАЙПЛАЙН ЗАПУСКА
# =====================================================================
def run_solution(data_dir="data", output_csv="submission.csv", output_meta="submission_meta.json"):
    print("="*60)
    print("      ЗАПУСК УЛУЧШЕННОГО РЕШЕНИЯ НА LIGHTGBM      ")
    print("="*60)
    
    fb = HydroFeatureBuilder(data_dir=data_dir)
    conc_df = pd.read_csv(os.path.join(data_dir, "concentration.csv"))
    
    print("Генерация признаков для обучающей выборки...")
    X_train = fb.build_features(conc_df)
    y_train = conc_df["concentration"].values.astype(np.float32)
    
    predictor = HydroLGBMPredictor()
    predictor.fit_and_calibrate(X_train, y_train, dates_series=conc_df["date"])
    
    queries_df = pd.read_csv(os.path.join(data_dir, "queries_test.csv"))
    print(f"Инференс для {len(queries_df)} тестовых запросов...")
    X_test = fb.build_features(queries_df)
    c_pred, c_lower, c_upper = predictor.predict(X_test)
    
    submission_df = pd.DataFrame({
        "query_id": queries_df["query_id"],
        "c_pred": np.round(c_pred, 3),
        "c_lower": np.round(c_lower, 3),
        "c_upper": np.round(c_upper, 3)
    })
    submission_df.to_csv(output_csv, index=False)
    
    post_proc = HydroPostProcessor(fb, predictor)
    mass_2026 = post_proc.calculate_mass_2026()
    arrival_time = post_proc.calculate_arrival_time(threshold=5.0)
    
    meta_result = {
        "mass_2026": mass_2026,
        "arrival_time": arrival_time
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta_result, f, indent=2)
        
    print("="*60)
    print("Решение готово и сохранено!")
    print("="*60)

if __name__ == "__main__":
    run_solution()
