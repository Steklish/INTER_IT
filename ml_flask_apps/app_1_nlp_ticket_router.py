"""
===============================================================================
FLASK USE CASE 1: ИНТЕЛЛЕКТУАЛЬНЫЙ NLP-МАРШРУТИЗАТОР ТИКЕТОВ ТЕХПОДДЕРЖКИ
===============================================================================

1. В ЧЕМ СУТЬ И ПОЧЕМУ ЭТО НЕ ТРИВИАЛЬНО?
----------------------------------------
Большинство учебных примеров ML работают только с числами (рост, вес, цена).
Но в реальном бизнесе 80% входящей информации — это НЕСТРУКТУРИРОВАННЫЙ ТЕКСТ:
обращения клиентов в поддержку, отзывы, резюме, юридические договоры.

Здесь мы создаем сквозной NLP-пайплайн (Natural Language Processing), который:
  1. Принимает сырой текст обращения клиента через веб-форму.
  2. Превращает слова в числа с помощью TF-IDF (Term Frequency - Inverse Document Frequency).
  3. Классифицирует обращение по 4 отделам компании:
     - 💳 Бухгалтерия и счета (Billing & Invoices)
     - 🛠️ Технические баги (Tech Support & Bugs)
     - 🔐 Безопасность и доступы (Security & Accounts)
     - 💼 Корпоративные продажи (Sales & Enterprise)
  4. Рассчитывает уверенность модели (Probability %) и рекомендуемый SLA (срок ответа).

2. КАК ЭТО РАБОТАЕТ В FLASK?
----------------------------
  - GET-запрос: пользователь открывает страницу в браузере -> сервер отдает HTML-форму.
  - POST-запрос: пользователь нажал кнопку "Отправить" -> Flask читает `request.form['ticket_text']`,
    прогоняет текст через `pipeline.predict_proba()` и обновляет HTML с результатами.

3. КАК ЗАПУСТИТЬ:
-----------------
    python app_1_nlp_ticket_router.py
Затем откройте в браузере: http://127.0.0.1:5001/
===============================================================================
"""

import os
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

# Компоненты Scikit-Learn для текста
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# НАСТРОЙКИ ПУТЕЙ И ПРИЛОЖЕНИЯ
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "nlp_ticket_pipeline.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


# =============================================================================
# ШАГ 1: ФУНКЦИЯ ДЛЯ АВТОМАТИЧЕСКОГО ОБУЧЕНИЯ МОДЕЛИ (ЕСЛИ ЕЕ ЕЩЕ НЕТ НА ДИСКЕ)
# =============================================================================
def train_and_save_nlp_pipeline():
    """
    Генерирует демонстрационную базу обращений в службу поддержки
    и обучает сквозной NLP-пайплайн.
    """
    print("--> [Обучение] Модель не найдена на диске. Обучаем NLP-пайплайн...")

    # Обучающая выборка: реальные примеры клиентских тикетов на русском и английском
    tickets = [
        # Бухгалтерия (Billing)
        ("I was charged twice on my credit card invoice this month, please refund", "Billing"),
        ("Где я могу скачать закрывающие акты и счет-фактуру за прошлый квартал?", "Billing"),
        ("My subscription renewed automatically, I want to cancel and get money back", "Billing"),
        ("Не проходит оплата банковской картой МИР, пишет ошибку транзакции", "Billing"),
        ("Need to change credit card details and update billing company address", "Billing"),
        ("Списали 49 долларов без предупреждения, требую немедленный возврат средств", "Billing"),
        ("Can you send me a receipt for the latest annual payment?", "Billing"),

        # Техническая поддержка (Tech Support)
        ("Application crashes with error 500 when exporting data to Excel", "Tech Support"),
        ("После последнего обновления кнопка сохранить не нажимается на странице", "Tech Support"),
        ("API endpoint returns 404 Not Found timeout exception during sync", "Tech Support"),
        ("Белый экран при входе в личный кабинет через браузер Chrome", "Tech Support"),
        ("Database connection drops every 10 minutes, high server memory usage", "Tech Support"),
        ("Мобильное приложение вылетает при открытии камеры на Android", "Tech Support"),
        ("Slow loading speed, dashboard takes over 30 seconds to fetch metrics", "Tech Support"),

        # Безопасность и доступ (Security)
        ("Cannot log into my account, 2FA SMS code never arrives to my phone", "Security"),
        ("Забыл пароль, а ссылка для восстановления на почту не приходит", "Security"),
        ("Suspicious login attempt detected from unknown IP address in Singapore", "Security"),
        ("Кажется, мой аккаунт взломали, изменился привязанный номер телефона", "Security"),
        ("Lost my Google Authenticator phone, need manual account verification", "Security"),
        ("Не могу отключить двухфакторную аутентификацию, требует старый телефон", "Security"),
        ("Unauthorized password reset request received, please lock my profile", "Security"),

        # Корпоративные продажи (Sales)
        ("We are a team of 150 people and want to negotiate an enterprise license", "Sales"),
        ("Хотим подключить 50 сотрудников, подскажите стоимость корпоративного тарифа", "Sales"),
        ("Can we schedule a product demo with your technical sales representative?", "Sales"),
        ("Интересует коммерческое предложение для крупного государственного заказчика", "Sales"),
        ("Do you offer discounts for annual educational university licenses?", "Sales"),
        ("Нужна презентация возможностей системы для совета директоров нашей компании", "Sales")
    ]

    df = pd.DataFrame(tickets, columns=['text', 'department'])

    # Создаем сквозной пайплайн:
    # 1. TfidfVectorizer:
    #    - ngram_range=(1, 2): смотрит как на отдельные слова ('error'), так и на пары слов ('error 500', 'get money')
    #    - sublinear_tf=True: сглаживает влияние слишком частых повторений одного и того же слова
    # 2. LogisticRegression:
    #    - Multi-class логистическая регрессия рассчитывает вероятность принадлежности к каждому из 4 отделов!
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ('classifier', LogisticRegression(C=1.5, random_state=42))
    ])

    pipeline.fit(df['text'], df['department'])

    # Сохраняем весь пайплайн (включая словарь слов и веса классификатора)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"--> [Готово] Пайплайн обучен и сохранен в: {MODEL_PATH}")
    return pipeline


# =============================================================================
# ШАГ 2: ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ПРИ СТАРТЕ СЕРВЕРА
# =============================================================================
if not MODEL_PATH.exists():
    nlp_model = train_and_save_nlp_pipeline()
else:
    print(f"--> Загружаем готовую модель из {MODEL_PATH}")
    nlp_model = joblib.load(MODEL_PATH)


# =============================================================================
# ШАГ 3: HTML-ШАБЛОН СТРАНИЦЫ (ВСТРОЕН ПРЯМО В КОД ДЛЯ ПРОСТОТЫ)
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>AI Support Ticket Router</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        h1 { color: #1e293b; margin-top: 0; font-size: 24px; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }
        .tagline { color: #64748b; font-size: 14px; margin-bottom: 25px; }
        label { font-weight: 600; color: #334155; display: block; margin-bottom: 8px; }
        textarea { width: 100%; box-sizing: border-box; height: 120px; padding: 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 15px; font-family: inherit; }
        textarea:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
        .btn-submit { background: #2563eb; color: #fff; border: none; padding: 12px 24px; font-size: 16px; font-weight: 600; border-radius: 6px; cursor: pointer; margin-top: 15px; }
        .btn-submit:hover { background: #1d4ed8; }
        .quick-samples { margin: 15px 0 25px; background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px dashed #cbd5e1; }
        .quick-btn { background: #e2e8f0; border: none; padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; margin-right: 6px; margin-bottom: 4px; }
        .quick-btn:hover { background: #cbd5e1; }
        
        .result-box { margin-top: 30px; border-radius: 8px; padding: 20px; border-left: 6px solid #2563eb; background: #f0fdf4; border-color: #16a34a; }
        .dept-title { font-size: 20px; font-weight: bold; color: #166534; margin: 0 0 10px; }
        .metric-row { display: flex; gap: 20px; margin: 15px 0; }
        .metric-card { background: #fff; padding: 12px 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; }
        .metric-val { font-size: 18px; font-weight: bold; color: #0f172a; }
        
        .probabilities { margin-top: 15px; }
        .prob-bar-container { background: #e2e8f0; border-radius: 4px; height: 12px; width: 100%; margin-top: 4px; overflow: hidden; }
        .prob-bar { background: #3b82f6; height: 100%; }
        .prob-row { margin-bottom: 10px; font-size: 13px; }
        .info-card { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; padding: 12px; border-radius: 6px; font-size: 13px; margin-top: 25px; }
    </style>
</head>
<body>

<div class="container">
    <h1>📨 AI Support Ticket Router (Маршрутизатор обращений)</h1>
    <div class="tagline">Пример NLP-пайплайна: TF-IDF токенизация + Мультиклассовая классификация в реальном времени</div>

    <form method="POST">
        <label for="ticket_text">Введите текст обращения клиента (на русском или английском):</label>
        
        <div class="quick-samples">
            <b>Быстрый тест (кликните, чтобы вставить пример):</b><br><br>
            <button type="button" class="quick-btn" onclick="setSample('Приложение падает с ошибкой 500 при экспорте отчета в Excel')">🐞 Баг (Tech Support)</button>
            <button type="button" class="quick-btn" onclick="setSample('С моей карты дважды списали деньги за тариф, требую вернуть средства!')">💳 Деньги (Billing)</button>
            <button type="button" class="quick-btn" onclick="setSample('Не приходит СМС с кодом двухфакторной аутентификации 2FA, не могу войти!')">🔐 Доступ (Security)</button>
            <button type="button" class="quick-btn" onclick="setSample('Мы крупный холдинг из 200 сотрудников, хотим получить коммерческое предложение на корпоративный тариф')">💼 Продажи (Sales)</button>
        </div>

        <textarea id="ticket_text" name="ticket_text" required placeholder="Напишите здесь проблему клиента...">{{ user_text if user_text else '' }}</textarea>
        <br>
        <button type="submit" class="btn-submit">⚡ Классифицировать и назначить отдел</button>
    </form>

    {% if result %}
    <div class="result-box">
        <div class="dept-title">Рекомендованный отдел: {{ result.department_ru }}</div>
        <div>Оригинальный класс: <code>{{ result.department_en }}</code></div>
        
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Уверенность модели</div>
                <div class="metric-val">{{ result.confidence }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Приоритет (SLA)</div>
                <div class="metric-val" style="color: {{ result.sla_color }};">{{ result.sla_text }}</div>
            </div>
        </div>

        <h4>Распределение вероятностей по всем отделам:</h4>
        <div class="probabilities">
            {% for dept, prob in result.all_probs.items() %}
            <div class="prob-row">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>{{ dept }}</b></span>
                    <span>{{ prob }}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: {{ prob }}%;"></div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <div class="info-card">
        <b>💡 Как это работает «под капотом»?</b><br>
        1. Введенный текст разбивается на N-граммы (слова и устойчивые словосочетания).<br>
        2. Формула TF-IDF взвешивает их информативность (слова вроде «ошибка 500» получают высокий вес, а «я», «в» — нулевой).<br>
        3. Линейная модель вычисляет вероятности через Softmax и выбирает наиболее вероятный отдел для маршрутизации тикета.
    </div>
</div>

<script>
function setSample(text) {
    document.getElementById('ticket_text').value = text;
}
</script>

</body>
</html>
"""

# Словарик для человекочитаемых названий отделов
DEPT_TRANSLATION = {
    'Billing': '💳 Бухгалтерия и расчеты (Billing & Invoices)',
    'Tech Support': '🛠️ Инженерная поддержка (Tech Support & Bugs)',
    'Security': '🔐 Служба безопасности и аккаунтов (Security & 2FA)',
    'Sales': '💼 Корпоративные продажи (Sales & Enterprise)'
}


# =============================================================================
# ШАГ 4: ОБРАБОТЧИК МАРШРУТОВ FLASK (GET И POST)
# =============================================================================
@app.route('/', methods=['GET', 'POST'])
def ticket_classifier():
    result = None
    user_text = ""

    # Если пользователь отправил форму методом POST
    if request.method == 'POST':
        user_text = request.form.get('ticket_text', '').strip()

        if user_text:
            # 1. Запуск сквозного NLP-инференса
            predicted_dept = nlp_model.predict([user_text])[0]
            probabilities = nlp_model.predict_proba([user_text])[0]
            classes = nlp_model.classes_

            # Словарь вероятностей для каждого отдела
            all_probs = {
                DEPT_TRANSLATION.get(c, c): round(p * 100, 1)
                for c, p in zip(classes, probabilities)
            }
            # Сортируем от максимальной вероятности к минимальной
            all_probs = dict(sorted(all_probs.items(), key=lambda item: item[1], reverse=True))

            confidence = round(max(probabilities) * 100, 1)

            # Эвристика для срока ответа (SLA)
            # Если это безопасность или высокий баг — приоритет КРИТИЧЕСКИЙ
            is_urgent_word = any(w in user_text.lower() for w in ['пароль', 'взлом', 'деньги', 'списали', 'crash', 'down', 'urgent', 'сразу'])
            if predicted_dept in ['Security', 'Billing'] or is_urgent_word:
                sla_text = "Критический (ответ до 1 часа)"
                sla_color = "#dc2626"
            elif predicted_dept == 'Tech Support':
                sla_text = "Высокий (ответ до 4 часов)"
                sla_color = "#ea580c"
            else:
                sla_text = "Стандартный (ответ в течение 24ч)"
                sla_color = "#16a34a"

            result = {
                'department_en': predicted_dept,
                'department_ru': DEPT_TRANSLATION.get(predicted_dept, predicted_dept),
                'confidence': confidence,
                'sla_text': sla_text,
                'sla_color': sla_color,
                'all_probs': all_probs
            }

    return render_template_string(HTML_TEMPLATE, result=result, user_text=user_text)


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 NLP Support Ticket Router запущен!")
    print("👉 Откройте в браузере: http://127.0.0.1:5001/")
    print("="*70 + "\n")
    app.run(debug=True, port=5001)
