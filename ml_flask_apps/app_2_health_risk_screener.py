"""
===============================================================================
FLASK USE CASE 2: ОБЪЯСНИМЫЙ МЕДИЦИНСКИЙ СКРИНИНГ СЕРДЕЧНО-СОСУДИСТОГО РИСКА
===============================================================================

1. В ЧЕМ СУТЬ И ПОЧЕМУ ЭТО НЕ ТРИВИАЛЬНО?
----------------------------------------
В медицине, кредитном скоринге и юриспруденции модели «черного ящика» (Black-Box),
которые просто говорят «Риск: ДА», ЗАПРЕЩЕНЫ или бесполезны. Врачу и пациенту
нужно знать: ПОЧЕМУ модель приняла такое решение, и ЧТО можно изменить?

В этом приложении реализовано:
  1. Оценка вероятности сердечно-сосудистого риска (0% - 100%).
  2. Интерпретация признаков (Feature Attribution): математическая декомпозиция
     вклада каждого фактора пациента (что повышает риск, а что защищает здоровье).
  3. «What-If» симулятор (Контрфактический анализ): расчет того, как изменится
     вероятность, если пациент бросит курить или снизит холестерин!

2. КАК ЭТО РАБОТАЕТ В FLASK?
----------------------------
Форма собирает клинические показатели (возраст, давление, холестерин, сигареты, спорт),
Flask передает их в пайплайн, модель вычисляет вероятности, раскладывает веса
на положительные и отрицательные вклады и генерирует персональные рекомендации.

3. КАК ЗАПУСТИТЬ:
-----------------
    python app_2_health_risk_screener.py
Затем откройте в браузере: http://127.0.0.1:5002/
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template_string

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "health_risk_pipeline.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ И ОБУЧЕНИЕ ИНТЕРПРЕТИРУЕМОЙ МОДЕЛИ
# =============================================================================
def train_and_save_health_model():
    print("--> [Обучение] Генерация медицинского датасета и обучение модели...")
    np.random.seed(42)
    n = 2000

    age = np.random.uniform(25, 80, size=n)
    blood_pressure = np.random.uniform(95, 190, size=n)
    cholesterol = np.random.uniform(130, 320, size=n)
    cigs_per_day = np.random.choice([0, 5, 10, 20, 30], p=[0.55, 0.15, 0.15, 0.10, 0.05], size=n)
    exercise_hours = np.random.uniform(0, 10, size=n)

    # Клиническая логика риска (на основе шкалы SCORE/Framingham):
    # Риск увеличивают: возраст, давление, холестерин, курение
    # Риск уменьшает: спорт (физическая активность)
    z = (
        -1.2 +
        0.05 * (age - 50) +
        0.035 * (blood_pressure - 120) +
        0.02 * (cholesterol - 190) +
        0.08 * cigs_per_day -
        0.30 * exercise_hours +
        np.random.normal(0, 0.4, size=n)
    )
    prob = 1 / (1 + np.exp(-z))
    high_risk = (prob >= 0.5).astype(int)

    df = pd.DataFrame({
        'age': np.round(age, 1),
        'blood_pressure': np.round(blood_pressure, 0),
        'cholesterol': np.round(cholesterol, 0),
        'cigs_per_day': cigs_per_day,
        'exercise_hours': np.round(exercise_hours, 1),
        'high_risk': high_risk
    })

    X = df[['age', 'blood_pressure', 'cholesterol', 'cigs_per_day', 'exercise_hours']]
    y = df['high_risk']

    # Пайплайн со стандартизацией:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"--> [Готово] Модель сохранена в: {MODEL_PATH}")
    return pipeline


if not MODEL_PATH.exists():
    health_model = train_and_save_health_model()
else:
    print(f"--> Загружаем готовую модель из {MODEL_PATH}")
    health_model = joblib.load(MODEL_PATH)


# =============================================================================
# ШАГ 2: HTML-ШАБЛОН С ФОРМОЙ И ВИЗУАЛЬНЫМ РАЗБОРОМ РИСКОВ
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Cardio Risk Screener</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f1f5f9; margin: 0; padding: 25px; }
        .container { max-width: 850px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }
        h1 { color: #0f172a; margin-top: 0; font-size: 24px; }
        .desc { color: #64748b; font-size: 14px; margin-bottom: 25px; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-weight: 600; color: #334155; margin-bottom: 6px; font-size: 14px; }
        .hint { font-size: 12px; color: #94a3b8; margin-top: 4px; }
        input[type="number"] { padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 15px; }
        input:focus { outline: none; border-color: #0284c7; box-shadow: 0 0 0 3px rgba(2,132,199,0.15); }
        .btn-calc { grid-column: span 2; background: #0284c7; color: #fff; border: none; padding: 14px; font-size: 16px; font-weight: bold; border-radius: 6px; cursor: pointer; margin-top: 10px; }
        .btn-calc:hover { background: #0369a1; }
        
        /* Карточка результата */
        .result-card { margin-top: 30px; border-radius: 8px; padding: 25px; background: #f8fafc; border: 1px solid #e2e8f0; }
        .risk-gauge { height: 18px; border-radius: 9px; background: #e2e8f0; overflow: hidden; margin: 15px 0; }
        .risk-fill { height: 100%; border-radius: 9px; transition: width 0.5s ease; }
        .risk-badge { display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px; color: #fff; }
        
        /* Таблица факторов */
        .factors-table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }
        .factors-table th { text-align: left; background: #f1f5f9; padding: 10px; border-bottom: 2px solid #cbd5e1; }
        .factors-table td { padding: 10px; border-bottom: 1px solid #e2e8f0; }
        .impact-bad { color: #dc2626; font-weight: 600; }
        .impact-good { color: #16a34a; font-weight: 600; }
        
        .what-if-box { margin-top: 25px; background: #fefce8; border: 1px solid #fef08a; padding: 15px; border-radius: 6px; color: #854d0e; }
    </style>
</head>
<body>

<div class="container">
    <h1>❤️ Объяснимый скрининг сердечно-сосудистого риска</h1>
    <div class="desc">Интерпретируемый ML: расчет индивидуальной вероятности и факторный анализ риска.</div>

    <form method="POST">
        <div class="form-grid">
            <div class="form-group">
                <label>Возраст (лет):</label>
                <input type="number" name="age" min="18" max="100" value="{{ form_data.age if form_data else 55 }}" required>
                <span class="hint">Нормальный диапазон: 25 - 80</span>
            </div>

            <div class="form-group">
                <label>Систолическое давление (мм рт. ст.):</label>
                <input type="number" name="blood_pressure" min="80" max="220" value="{{ form_data.blood_pressure if form_data else 145 }}" required>
                <span class="hint">Норма: 110 - 125, высокое: &gt; 140</span>
            </div>

            <div class="form-group">
                <label>Общий холестерин (мг/дл):</label>
                <input type="number" name="cholesterol" min="100" max="400" value="{{ form_data.cholesterol if form_data else 240 }}" required>
                <span class="hint">Норма: &lt; 200, повышенный: &gt; 239</span>
            </div>

            <div class="form-group">
                <label>Выкуриваемых сигарет в день:</label>
                <input type="number" name="cigs_per_day" min="0" max="60" value="{{ form_data.cigs_per_day if form_data else 15 }}" required>
                <span class="hint">0 — не курит</span>
            </div>

            <div class="form-group" style="grid-column: span 2;">
                <label>Физическая активность (часов спорта в неделю):</label>
                <input type="number" name="exercise_hours" min="0" max="25" step="0.5" value="{{ form_data.exercise_hours if form_data else 1.0 }}" required>
                <span class="hint">Рекомендация ВОЗ: от 2.5 до 5 часов умеренной активности в неделю</span>
            </div>

            <button type="submit" class="btn-calc">📊 Рассчитать риск и показать объяснение факторов</button>
        </div>
    </form>

    {% if result %}
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="margin: 0;">Оценка риска сердечно-сосудистых осложнений</h2>
            <span class="risk-badge" style="background: {{ result.badge_color }};">{{ result.risk_category }}</span>
        </div>

        <div style="font-size: 28px; font-weight: bold; margin: 15px 0 5px; color: {{ result.badge_color }};">
            {{ result.risk_pct }}%
        </div>

        <div class="risk-gauge">
            <div class="risk-fill" style="width: {{ result.risk_pct }}%; background: {{ result.badge_color }};"></div>
        </div>

        <h3>🔍 Декомпозиция факторов (Почему модель поставила такой балл?):</h3>
        <table class="factors-table">
            <thead>
                <tr>
                    <th>Фактор здоровья</th>
                    <th>Значение пациента</th>
                    <th>Влияние на риск</th>
                    <th>Интерпретация</th>
                </tr>
            </thead>
            <tbody>
                {% for f in result.factor_breakdown %}
                <tr>
                    <td><b>{{ f.name }}</b></td>
                    <td>{{ f.value }}</td>
                    <td class="{{ 'impact-bad' if f.impact_num > 0 else 'impact-good' }}">
                        {{ '+' if f.impact_num > 0 else '' }}{{ f.impact_str }}
                    </td>
                    <td>{{ f.note }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        {% if result.what_if %}
        <div class="what-if-box">
            <b>💡 Контрфактический анализ («Что если изменится образ жизни?»):</b><br>
            {{ result.what_if }}
        </div>
        {% endif %}
    </div>
    {% endif %}
</div>

</body>
</html>
"""


# =============================================================================
# ШАГ 3: ОБРАБОТЧИК ЗАПРОСОВ И РАСЧЕТ ИНТЕРПРЕТАЦИИ
# =============================================================================
@app.route('/', methods=['GET', 'POST'])
def health_screener():
    result = None
    form_data = None

    if request.method == 'POST':
        # Считываем входные параметры из формы
        age = float(request.form.get('age', 50))
        blood_pressure = float(request.form.get('blood_pressure', 120))
        cholesterol = float(request.form.get('cholesterol', 190))
        cigs_per_day = float(request.form.get('cigs_per_day', 0))
        exercise_hours = float(request.form.get('exercise_hours', 2.0))

        form_data = {
            'age': age,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'cigs_per_day': cigs_per_day,
            'exercise_hours': exercise_hours
        }

        # Оборачиваем в DataFrame с точными именами признаков
        patient_df = pd.DataFrame([form_data])

        # 1. Основное предсказание модели
        prob = health_model.predict_proba(patient_df)[0][1]
        risk_pct = round(prob * 100, 1)

        # Категория риска
        if risk_pct < 20:
            risk_category = "НИЗКИЙ РИСК"
            badge_color = "#16a34a"  # Зеленый
        elif risk_pct < 50:
            risk_category = "УМЕРЕННЫЙ РИСК"
            badge_color = "#d97706"  # Желто-оранжевый
        else:
            risk_category = "ВЫСОКИЙ РИСК"
            badge_color = "#dc2626"  # Красный

        # 2. Математическая декомпозиция вклада каждого фактора:
        # Извлекаем обученный скейлер и веса логистической регрессии
        scaler = health_model.named_steps['scaler']
        clf = health_model.named_steps['classifier']

        # Вектор стандартизированных признаков пациента z = (x - mean) / scale
        patient_scaled = scaler.transform(patient_df)[0]
        weights = clf.coef_[0]

        # Вклад каждого признака в log-odds = weight * scaled_value
        contributions = weights * patient_scaled

        factor_names = [
            ("Возраст", f"{int(age)} лет"),
            ("Артериальное давление", f"{int(blood_pressure)} мм рт. ст."),
            ("Общий холестерин", f"{int(cholesterol)} мг/дл"),
            ("Курение", f"{int(cigs_per_day)} сиг./день"),
            ("Физическая активность", f"{exercise_hours} ч/неделю")
        ]

        factor_breakdown = []
        for (name, val_str), contrib in zip(factor_names, contributions):
            impact_num = round(contrib, 2)
            if impact_num > 0.2:
                note = "Существенно повышает нагрузку на сосуды"
            elif impact_num > 0.05:
                note = "Незначительно увеличивает риск"
            elif impact_num < -0.2:
                note = "Мощный защитный фактор сердечно-сосудистой системы"
            else:
                note = "В пределах возрастной нормы"

            factor_breakdown.append({
                'name': name,
                'value': val_str,
                'impact_num': impact_num,
                'impact_str': f"{impact_num:+.2f} pts к риску",
                'note': note
            })

        # 3. What-If анализ: Что будет, если пациент оптимизирует образ жизни?
        # Симулируем: бросил курить (cigs=0), давление снижено до 125, холестерин до 190, спорт 3.5 ч.
        improved_df = patient_df.copy()
        improved_df['cigs_per_day'] = 0.0
        improved_df['blood_pressure'] = min(blood_pressure, 125.0)
        improved_df['cholesterol'] = min(cholesterol, 190.0)
        improved_df['exercise_hours'] = max(exercise_hours, 3.5)

        improved_prob = health_model.predict_proba(improved_df)[0][1]
        improved_pct = round(improved_prob * 100, 1)

        diff = round(risk_pct - improved_pct, 1)
        if diff > 3:
            what_if = (
                f"Если снизить давление до нормы (125), нормализовать холестерин, "
                f"отказаться от курения и заниматься спортом от 3.5 ч/нед, расчетный риск упадет с "
                f"<b>{risk_pct}%</b> до <b>{improved_pct}%</b> (снижение на {diff}% пунктов)!"
            )
        else:
            what_if = "Показатели пациента уже близки к оптимальным. Рекомендуется поддерживать текущий режим!"

        result = {
            'risk_pct': risk_pct,
            'risk_category': risk_category,
            'badge_color': badge_color,
            'factor_breakdown': factor_breakdown,
            'what_if': what_if
        }

    return render_template_string(HTML_TEMPLATE, result=result, form_data=form_data)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Explainable Health Risk Screener запущен!")
    print("👉 Откройте в браузере: http://127.0.0.1:5002/")
    print("="*70 + "\n")
    app.run(debug=True, port=5002)
