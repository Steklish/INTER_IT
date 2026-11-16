"""
===============================================================================
FLASK USE CASE 3: ДИНАМИЧЕСКАЯ ОЦЕНКА НЕДВИЖИМОСТИ И СИМУЛЯТОР РЕМОНТА (ROI)
===============================================================================

1. В ЧЕМ СУТЬ И ПОЧЕМУ ЭТО НЕ ТРИВИАЛЬНО?
----------------------------------------
В реальном бизнесе (FinTech, банки при выдаче ипотеки, сервисы объявлений вроде Циан/Zillow)
оценка квартиры должна:
  1. Работать со смешанными типами данных (числа: площадь, год постройки; текст: район, тип дома).
  2. Использовать сквозной ColumnTransformer без утечки данных.
  3. Не просто выдать «голую цифру», а рассчитать:
     - Оценочную стоимость и диапазон рыночной погрешности (+/- 7%).
     - Стоимость квадратного метра.
     - Инвестиционную привлекательность ремонта (ROI): «Что будет, если сделать
       дизайнерский ремонт вместо косметического? Сколько это добавит к цене?».

2. КАК ЭТО РАБОТАЕТ В FLASK?
----------------------------
Форма содержит селекторы районов, типов домов и ползунки метража.
Flask формирует DataFrame с правильными типами данных, пропускает через пайплайн
с градиентным бустингом (HistGradientBoosting / LightGBM) и выполняет два прогона:
  - Прогон 1: Оценка текущего состояния объекта.
  - Прогон 2: Оценка объекта с улучшенным ремонтом (симуляция добавленной стоимости).

3. КАК ЗАПУСТИТЬ:
-----------------
    python app_3_dynamic_pricing.py
Затем откройте в браузере: http://127.0.0.1:5003/
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template_string

# Компоненты Scikit-Learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "real_estate_pipeline.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ И ОБУЧЕНИЕ СЛОЖНОГО ПАЙПЛАЙНА
# =============================================================================
def train_and_save_pricing_model():
    print("--> [Обучение] Генерация данных о недвижимости и обучение градиентного бустинга...")
    np.random.seed(42)
    n = 2500

    districts = ['Central', 'Business Bay', 'University Hub', 'Green Suburbs', 'Industrial West']
    building_types = ['Monolith', 'Brick', 'Panel']
    renovations = ['Needs Renovation', 'Cosmetic', 'Euro / Designer']

    dist = np.random.choice(districts, p=[0.15, 0.20, 0.25, 0.25, 0.15], size=n)
    b_type = np.random.choice(building_types, p=[0.4, 0.35, 0.25], size=n)
    renov = np.random.choice(renovations, p=[0.3, 0.45, 0.25], size=n)

    area = np.random.uniform(28, 140, size=n)
    rooms = np.clip((area / 30).astype(int) + np.random.randint(-1, 2, size=n), 1, 5)
    year_built = np.random.randint(1970, 2024, size=n)
    floor = np.random.randint(1, 26, size=n)
    metro_min = np.random.uniform(2, 35, size=n)

    # Рыночная стоимость:
    district_mult = {
        'Central': 1.6, 'Business Bay': 1.35, 'University Hub': 1.1,
        'Green Suburbs': 0.95, 'Industrial West': 0.8
    }
    renov_bonus = {
        'Needs Renovation': 0.0, 'Cosmetic': 18000.0, 'Euro / Designer': 45000.0
    }
    b_type_bonus = {'Monolith': 15000.0, 'Brick': 8000.0, 'Panel': 0.0}

    base_sqm_price = 1400.0  # базовая цена кв.м

    prices = []
    for d, b, r, a, y, m in zip(dist, b_type, renov, area, year_built, metro_min):
        sqm_cost = base_sqm_price * district_mult[d]
        # Новые дома дороже
        age_bonus = max(0, (y - 1980) * 150)
        # Близость к метро повышает цену
        metro_penalty = (m - 5) * 600
        noise = np.random.normal(0, 8000)

        total = (a * sqm_cost) + renov_bonus[r] + b_type_bonus[b] + age_bonus - metro_penalty + noise
        prices.append(max(total, 35000))

    df = pd.DataFrame({
        'district': dist,
        'building_type': b_type,
        'renovation': renov,
        'area_sqm': np.round(area, 1),
        'rooms': rooms,
        'year_built': year_built,
        'floor': floor,
        'metro_min': np.round(metro_min, 0),
        'price_usd': np.round(prices, 2)
    })

    X = df.drop(columns=['price_usd'])
    y = df['price_usd']

    numeric_cols = ['area_sqm', 'rooms', 'year_built', 'floor', 'metro_min']
    categorical_cols = ['district', 'building_type', 'renovation']

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', HistGradientBoostingRegressor(max_iter=200, random_state=42))
    ])

    full_pipeline.fit(X, y)
    joblib.dump(full_pipeline, MODEL_PATH)
    print(f"--> [Готово] Модель сохранена в: {MODEL_PATH}")
    return full_pipeline


if not MODEL_PATH.exists():
    pricing_model = train_and_save_pricing_model()
else:
    print(f"--> Загружаем готовую модель из {MODEL_PATH}")
    pricing_model = joblib.load(MODEL_PATH)


# =============================================================================
# ШАГ 2: HTML-ШАБЛОН СТРАНИЦЫ
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Real Estate Price Evaluator</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 25px; }
        .container { max-width: 850px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }
        h1 { color: #0f172a; margin-top: 0; font-size: 24px; }
        .subtitle { color: #64748b; font-size: 14px; margin-bottom: 25px; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        label { font-weight: 600; color: #334155; font-size: 13px; margin-bottom: 6px; display: block; }
        input, select { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
        .btn-calc { grid-column: span 2; background: #059669; color: #fff; border: none; padding: 14px; font-size: 16px; font-weight: bold; border-radius: 6px; cursor: pointer; margin-top: 10px; }
        .btn-calc:hover { background: #047857; }
        
        .result-box { margin-top: 30px; padding: 25px; border-radius: 8px; background: #f0fdf4; border: 1px solid #bbf7d0; }
        .main-price { font-size: 32px; font-weight: 800; color: #15803d; margin: 10px 0; }
        .price-stats { display: flex; gap: 20px; margin: 15px 0; }
        .stat-card { background: #fff; padding: 12px 18px; border-radius: 6px; border: 1px solid #dcfce7; }
        .stat-label { font-size: 11px; text-transform: uppercase; color: #64748b; }
        .stat-val { font-size: 16px; font-weight: bold; color: #1e293b; }
        
        .renovation-roi { margin-top: 20px; background: #eff6ff; border: 1px solid #bfdbfe; padding: 18px; border-radius: 8px; color: #1e40af; }
        .roi-title { font-weight: bold; font-size: 16px; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
    </style>
</head>
<body>

<div class="container">
    <h1>🏢 Калькулятор рыночной стоимости недвижимости & ROI ремонта</h1>
    <div class="subtitle">Сквозной пайплайн Scikit-Learn: численные + категориальные признаки + What-If симуляция</div>

    <form method="POST">
        <div class="form-grid">
            <div>
                <label>Район города:</label>
                <select name="district">
                    <option value="Central" {{ 'selected' if form.district == 'Central' else '' }}>Central (Исторический центр)</option>
                    <option value="Business Bay" {{ 'selected' if form.district == 'Business Bay' else '' }}>Business Bay (Деловой квартал)</option>
                    <option value="University Hub" {{ 'selected' if form.district == 'University Hub' else '' }}>University Hub (Студенческий кластер)</option>
                    <option value="Green Suburbs" {{ 'selected' if form.district == 'Green Suburbs' else '' }}>Green Suburbs (Спальный эко-район)</option>
                    <option value="Industrial West" {{ 'selected' if form.district == 'Industrial West' else '' }}>Industrial West (Промзона)</option>
                </select>
            </div>

            <div>
                <label>Тип дома:</label>
                <select name="building_type">
                    <option value="Monolith" {{ 'selected' if form.building_type == 'Monolith' else '' }}>Монолитный</option>
                    <option value="Brick" {{ 'selected' if form.building_type == 'Brick' else '' }}>Кирпичный</option>
                    <option value="Panel" {{ 'selected' if form.building_type == 'Panel' else '' }}>Панельный</option>
                </select>
            </div>

            <div>
                <label>Площадь (кв. м):</label>
                <input type="number" name="area_sqm" step="0.5" min="20" max="250" value="{{ form.area_sqm if form else 65.0 }}" required>
            </div>

            <div>
                <label>Количество комнат:</label>
                <input type="number" name="rooms" min="1" max="6" value="{{ form.rooms if form else 2 }}" required>
            </div>

            <div>
                <label>Год постройки дома:</label>
                <input type="number" name="year_built" min="1950" max="2025" value="{{ form.year_built if form else 2018 }}" required>
            </div>

            <div>
                <label>Этаж:</label>
                <input type="number" name="floor" min="1" max="40" value="{{ form.floor if form else 7 }}" required>
            </div>

            <div>
                <label>Минут пешком до метро:</label>
                <input type="number" name="metro_min" min="1" max="60" value="{{ form.metro_min if form else 8 }}" required>
            </div>

            <div>
                <label>Текущее состояние ремонта:</label>
                <select name="renovation">
                    <option value="Needs Renovation" {{ 'selected' if form.renovation == 'Needs Renovation' else '' }}>Требует ремонта (Черновая)</option>
                    <option value="Cosmetic" {{ 'selected' if form.renovation == 'Cosmetic' else '' }}>Косметический ремонт</option>
                    <option value="Euro / Designer" {{ 'selected' if form.renovation == 'Euro / Designer' else '' }}>Евроремонт / Дизайнерский</option>
                </select>
            </div>

            <button type="submit" class="btn-calc">📈 Рассчитать стоимость и симуляцию ROI</button>
        </div>
    </form>

    {% if result %}
    <div class="result-box">
        <div style="font-size: 13px; color: #166534; font-weight: 600;">ОЦЕНКА РЫНОЧНОЙ СТОИМОСТИ ОБЪЕКТА</div>
        <div class="main-price">${{ "{:,.0f}".format(result.estimated_price) }}</div>
        
        <div class="price-stats">
            <div class="stat-card">
                <div class="stat-label">Диапазон рынка (±7%)</div>
                <div class="stat-val">${{ "{:,.0f}".format(result.price_min) }} — ${{ "{:,.0f}".format(result.price_max) }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Цена за 1 кв.м</div>
                <div class="stat-val">${{ "{:,.0f}".format(result.price_per_sqm) }} / м²</div>
            </div>
        </div>

        <div class="renovation-roi">
            <div class="roi-title">🔨 Инвестиционная симуляция ремонта (What-If Analysis):</div>
            <div>{{ result.roi_text | safe }}</div>
        </div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""


# =============================================================================
# ШАГ 3: ОБРАБОТЧИК ЗАПРОСОВ
# =============================================================================
@app.route('/', methods=['GET', 'POST'])
def real_estate_evaluator():
    result = None
    form = {
        'district': 'Business Bay',
        'building_type': 'Monolith',
        'renovation': 'Cosmetic',
        'area_sqm': 65.0,
        'rooms': 2,
        'year_built': 2018,
        'floor': 7,
        'metro_min': 8
    }

    if request.method == 'POST':
        form = {
            'district': request.form.get('district', 'Business Bay'),
            'building_type': request.form.get('building_type', 'Monolith'),
            'renovation': request.form.get('renovation', 'Cosmetic'),
            'area_sqm': float(request.form.get('area_sqm', 65.0)),
            'rooms': int(request.form.get('rooms', 2)),
            'year_built': int(request.form.get('year_built', 2018)),
            'floor': int(request.form.get('floor', 7)),
            'metro_min': float(request.form.get('metro_min', 8.0))
        }

        # Создаем датафрейм для инференса
        input_df = pd.DataFrame([form])

        # 1. Основное предсказание
        estimated_price = pricing_model.predict(input_df)[0]
        price_min = estimated_price * 0.93
        price_max = estimated_price * 1.07
        price_per_sqm = estimated_price / form['area_sqm']

        # 2. What-If симуляция: Что даст апгрейд ремонта?
        current_renov = form['renovation']
        if current_renov != 'Euro / Designer':
            simulated_df = input_df.copy()
            simulated_df['renovation'] = 'Euro / Designer'
            simulated_price = pricing_model.predict(simulated_df)[0]
            price_delta = simulated_price - estimated_price

            # Примерная стоимость ремонта $300 за кв.м
            approx_renovation_cost = form['area_sqm'] * 300
            net_profit = price_delta - approx_renovation_cost

            roi_text = (
                f"Если сделать <b>Евроремонт / Дизайнерский</b>, рыночная цена квартиры поднимется "
                f"до <b>${simulated_price:,.0f}</b> (+<b>${price_delta:,.0f}</b> к капитализации).<br>"
                f"Ориентировочные затраты на материалы и работы (~$300/м²): <b>${approx_renovation_cost:,.0f}</b>.<br>"
                f"Чистая прибыль (Net Profit): <b style='color: {'#16a34a' if net_profit > 0 else '#dc2626'}'>"
                f"{'+' if net_profit > 0 else ''}${net_profit:,.0f}</b>."
            )
        else:
            roi_text = "В объекте уже выполнен премиальный ремонт (Euro / Designer). Капитализация максимальна!"

        result = {
            'estimated_price': estimated_price,
            'price_min': price_min,
            'price_max': price_max,
            'price_per_sqm': price_per_sqm,
            'roi_text': roi_text
        }

    return render_template_string(HTML_TEMPLATE, result=result, form=form)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Dynamic Real Estate Evaluator запущен!")
    print("👉 Откройте в браузере: http://127.0.0.1:5003/")
    print("="*70 + "\n")
    app.run(debug=True, port=5003)
