"""
===============================================================================
FLASK MASTER PORTAL: ЕДИНЫЙ ПОРТАЛ ВСЕХ 4 ML-ПРИЛОЖЕНИЙ
===============================================================================

Запустив этот один файл, вы получаете доступ ко ВСЕМ четырем приложениям
через единую удобную панель навигации:
  - Вкладка 1: 📨 NLP-маршрутизатор обращений клиентов
  - Вкладка 2: ❤️ Объяснимый медицинский риск (Feature Attribution)
  - Вкладка 3: 🏢 Динамическая оценка недвижимости и ROI ремонта
  - Вкладка 4: 🛡️ Пакетный аудит реестра транзакций (CSV Upload & Isolation Forest)

ЗАПУСК:
    python portal.py
Затем откройте в браузере: http://127.0.0.1:5000/
===============================================================================
"""

import os
from pathlib import Path
from flask import Flask, render_template_string, redirect, url_for

# Импортируем готовые обработчики и приложения
import app_1_nlp_ticket_router as app1
import app_2_health_risk_screener as app2
import app_3_dynamic_pricing as app3
import app_4_fraud_detector as app4
import app_5_image_feature_gbm as app5
import app_6_ocr_pipeline as app6

app = Flask(__name__)

# Общий шаблон шапки (Navigation Bar)
NAV_HEADER = """
<div style="background: #0f172a; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; color: #fff; font-family: -apple-system, sans-serif;">
    <div style="font-size: 18px; font-weight: bold;">
        <a href="/" style="color: #38bdf8; text-decoration: none;">🚀 Machine Learning Flask Showcase</a>
    </div>
    <div style="display: flex; gap: 15px;">
        <a href="/app1" style="color: #e2e8f0; text-decoration: none; font-size: 14px; padding: 6px 12px; border-radius: 4px; background: #1e293b;">📨 1. NLP Router</a>
        <a href="/app2" style="color: #e2e8f0; text-decoration: none; font-size: 14px; padding: 6px 12px; border-radius: 4px; background: #1e293b;">❤️ 2. Cardio Risk</a>
        <a href="/app3" style="color: #e2e8f0; text-decoration: none; font-size: 14px; padding: 6px 12px; border-radius: 4px; background: #1e293b;">🏢 3. Real Estate</a>
        <a href="/app4" style="color: #e2e8f0; text-decoration: none; font-size: 14px; padding: 6px 12px; border-radius: 4px; background: #1e293b;">🛡️ 4. Fraud Audit</a>
        <a href="/app5" style="color: #e2e8f0; text-decoration: none; font-size: 14px; padding: 6px 12px; border-radius: 4px; background: #1e293b;">🖼️ 5. Vision + GBM</a>
        <a href="/app6" style="color: #e2e8f0; text-decoration: none; font-size: 14px; padding: 6px 12px; border-radius: 4px; background: #1e293b;">📄 6. OCR Pipeline</a>
    </div>
</div>
"""

HOME_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>ML Flask Showcase Portal</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 0; }}
        .content {{ max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #0f172a; font-size: 28px; margin-bottom: 10px; }}
        .lead {{ color: #475569; font-size: 16px; margin-bottom: 35px; }}
        .cards-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }}
        .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.08); }}
        .card-icon {{ font-size: 32px; margin-bottom: 15px; }}
        .card-title {{ font-size: 20px; font-weight: bold; color: #1e293b; margin-bottom: 8px; }}
        .card-desc {{ font-size: 14px; color: #64748b; line-height: 1.5; margin-bottom: 20px; min-height: 60px; }}
        .card-tech {{ background: #f1f5f9; padding: 6px 12px; border-radius: 4px; font-size: 12px; color: #334155; margin-bottom: 20px; }}
        .card-btn {{ display: inline-block; background: #2563eb; color: #fff; text-decoration: none; padding: 10px 18px; border-radius: 6px; font-weight: 600; font-size: 14px; }}
        .card-btn:hover {{ background: #1d4ed8; }}
    </style>
</head>
<body>

{NAV_HEADER}

<div class="content">
    <h1>Витрина нетривиальных сценариев машинного обучения на Flask</h1>
    <p class="lead">Все модели упакованы в сквозные пайплайны и обучены на реалистичных задачах бизнеса:</p>

    <div class="cards-grid">
        <!-- Карточка 1 -->
        <div class="card">
            <div class="card-icon">📨</div>
            <div class="card-title">1. NLP Маршрутизатор обращений</div>
            <div class="card-desc">Принимает сырой текст пользователя, векторизует через TF-IDF и автоматически направляет в один из 4 отделов компании, рассчитывая SLA и вероятность.</div>
            <div class="card-tech"><b>Стек:</b> TfidfVectorizer + Multiclass Logistic Regression</div>
            <a href="/app1" class="card-btn">Открыть приложение →</a>
        </div>

        <!-- Карточка 2 -->
        <div class="card">
            <div class="card-icon">❤️</div>
            <div class="card-title">2. Объяснимый медицинский риск</div>
            <div class="card-desc">Не «черный ящик»: показывает не только % риска, но и точный математический вклад каждого фактора (давление, холестерин, курение) и «What-If» симуляцию.</div>
            <div class="card-tech"><b>Стек:</b> StandardScaler + Logistic Regression Feature Attribution</div>
            <a href="/app2" class="card-btn">Открыть приложение →</a>
        </div>

        <!-- Карточка 3 -->
        <div class="card">
            <div class="card-icon">🏢</div>
            <div class="card-title">3. Оценка недвижимости & ROI ремонта</div>
            <div class="card-desc">Смешанные типы признаков (районы, типы домов, метраж). Предсказывает капитализацию и мгновенно симулирует финансовую выгоду от евроремонта.</div>
            <div class="card-tech"><b>Стек:</b> ColumnTransformer + HistGradientBoostingRegressor</div>
            <a href="/app3" class="card-btn">Открыть приложение →</a>
        </div>

        <!-- Карточка 4 -->
        <div class="card">
            <div class="card-icon">🛡️</div>
            <div class="card-title">4. Пакетный аудит и поиск фрода</div>
            <div class="card-desc">Загрузка файла CSV с сотнями транзакций. Обучение без учителя (Isolation Forest) изолирует аномальные операции и выводит отчет с подсветкой.</div>
            <div class="card-tech"><b>Стек:</b> RobustScaler + Isolation Forest + CSV File Stream</div>
            <a href="/app4" class="card-btn">Открыть приложение →</a>
        </div>

        <!-- Карточка 5 -->
        <div class="card">
            <div class="card-icon">🖼️</div>
            <div class="card-title">5. Компьютерное зрение + LightGBM</div>
            <div class="card-desc">Гибридный инференс: предобученная MobileNetV3 извлекает 576 признаков изображения, а L2 + PCA + LightGBM молниеносно классифицируют тип ландшафта с 2D проекцией EDA.</div>
            <div class="card-tech"><b>Стек:</b> PyTorch MobileNetV3 + L2 Normalizer + PCA + LightGBM</div>
            <a href="/app5" class="card-btn">Открыть приложение →</a>
        </div>

        <!-- Карточка 6 -->
        <div class="card">
            <div class="card-icon">📄</div>
            <div class="card-title">6. Интеллектуальный OCR-пайплайн</div>
            <div class="card-desc">Оптическое распознавание документов: адаптивная бинаризация OpenCV, разметка рамок слов (Bounding Boxes), LSTM-чтение текста и извлечение сущностей (RegEx / NLP).</div>
            <div class="card-tech"><b>Стек:</b> OpenCV + Tesseract OCR Engine + Entity Extractor</div>
            <a href="/app6" class="card-btn">Открыть приложение →</a>
        </div>
    </div>
</div>

</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)


# Прокси-маршруты к каждому приложению с сохранением верхней панели навигации
@app.route('/app1', methods=['GET', 'POST'])
def view_app1():
    response = app1.ticket_classifier()
    if isinstance(response, str):
        return NAV_HEADER + response
    return response


@app.route('/app2', methods=['GET', 'POST'])
def view_app2():
    response = app2.health_screener()
    if isinstance(response, str):
        return NAV_HEADER + response
    return response


@app.route('/app3', methods=['GET', 'POST'])
def view_app3():
    response = app3.real_estate_evaluator()
    if isinstance(response, str):
        return NAV_HEADER + response
    return response


@app.route('/app4', methods=['GET', 'POST'])
def view_app4():
    response = app4.fraud_audit()
    if isinstance(response, str):
        return NAV_HEADER + response
    return response


@app.route('/app5', methods=['GET', 'POST'])
def view_app5():
    response = app5.vision_classifier()
    if isinstance(response, str):
        return NAV_HEADER + response
    return response


@app.route('/app6', methods=['GET', 'POST'])
def view_app6():
    response = app6.ocr_view()
    if isinstance(response, str):
        return NAV_HEADER + response
    return response


@app.route('/download_sample')
def download_sample():
    return app4.download_sample()


if __name__ == '__main__':
    print("\n" + "="*75)
    print("🌟 ЕДИНЫЙ ПОРТАЛ ВСЕХ ML-ПРИЛОЖЕНИЙ ЗАПУЩЕН!")
    print("👉 Откройте главную страницу в браузере: http://127.0.0.1:5000/")
    print("="*75 + "\n")
    app.run(debug=True, port=5000)
