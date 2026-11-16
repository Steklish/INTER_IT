"""
===============================================================================
FLASK USE CASE 7: ИНТЕРАКТИВНЫЙ ТРАНСФОРМЕР (EMBEDDINGS + CLASSIFICATION)
===============================================================================

1. В ЧЕМ СУТЬ?
--------------
Приложение демонстрирует локальный запуск компактного трансформера
(sentence-transformers/all-MiniLM-L6-v2) на CPU:
  1. Извлечение 384-мерного контекстного семантического эмбеддинга.
  2. Инспекция токенизации WordPiece (токены [CLS], [SEP], субслова).
  3. Классификация интента / тональности (Positive, Billing, Technical, Security).
  4. Демонстрация Matryoshka Representation Learning (MRL) — усечение вектора до 64 dims.

2. КАК ЗАПУСТИТЬ:
-----------------
    python app_7_transformer_pipeline.py
Затем откройте в браузере: http://127.0.0.1:5007/
===============================================================================
"""

import os
from pathlib import Path
from flask import Flask, request, render_template_string

from transformer_pipeline import TransformerClassifierPipeline

app = Flask(__name__)
pipeline = TransformerClassifierPipeline()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Local Transformer NLP Pipeline</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 25px; }
        .container { max-width: 950px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }
        h1 { color: #0f172a; margin-top: 0; font-size: 24px; }
        .desc { color: #64748b; font-size: 14px; margin-bottom: 25px; line-height: 1.5; }
        
        label { font-weight: 600; color: #334155; display: block; margin-bottom: 8px; font-size: 14px; }
        textarea { width: 100%; box-sizing: border-box; height: 110px; padding: 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 15px; font-family: inherit; }
        textarea:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.15); }
        .btn-run { background: #4f46e5; color: #fff; border: none; padding: 12px 24px; font-size: 15px; font-weight: bold; border-radius: 6px; cursor: pointer; margin-top: 12px; }
        .btn-run:hover { background: #4338ca; }
        
        .samples-box { background: #f1f5f9; padding: 12px; border-radius: 6px; margin-bottom: 15px; font-size: 13px; }
        .sample-btn { background: #fff; border: 1px solid #cbd5e1; padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; margin: 4px 4px 0 0; }
        .sample-btn:hover { border-color: #4f46e5; background: #eef2ff; color: #4338ca; }
        
        .result-box { margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 25px; }
        .top-badge { display: inline-block; padding: 6px 14px; border-radius: 20px; background: #e0e7ff; color: #3730a3; font-weight: bold; font-size: 15px; margin-bottom: 15px; }
        
        .prob-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }
        .prob-table th { text-align: left; background: #f8fafc; padding: 10px; border-bottom: 2px solid #cbd5e1; }
        .prob-table td { padding: 10px; border-bottom: 1px solid #f1f5f9; }
        .prob-bar-bg { background: #e2e8f0; border-radius: 4px; height: 8px; width: 100%; margin-top: 4px; overflow: hidden; }
        .prob-bar-fill { background: #4f46e5; height: 100%; }
        
        .token-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
        .token-pill { background: #f1f5f9; border: 1px solid #cbd5e1; padding: 4px 8px; border-radius: 4px; font-family: monospace; font-size: 12px; color: #1e293b; }
        .token-pill.special { background: #fef3c7; border-color: #fcd34d; color: #92400e; font-weight: bold; }
        
        .vector-box { background: #0f172a; color: #38bdf8; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 12px; margin-top: 10px; overflow-x: auto; white-space: nowrap; }
        .info-panel { margin-top: 30px; background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; padding: 16px; border-radius: 8px; font-size: 13px; line-height: 1.6; }
    </style>
</head>
<body>

<div class="container">
    <h1>🤖 Локальный Трансформер: Извлечение эмбеддингов & Классификация</h1>
    <div class="desc">
        Запуск компактной модели <b>sentence-transformers/all-MiniLM-L6-v2 (~80 МБ)</b> на процессоре (CPU). 
        Трансформер сжимает смысл предложения в плотный 384-мерный вектор через механизм Self-Attention и Mean Pooling.
    </div>

    <form method="POST">
        <div class="samples-box">
            <b>Быстрый тест (кликните, чтобы вставить пример):</b><br>
            <button type="button" class="sample-btn" onclick="setSample('Отличная платформа, техническая поддержка ответила за 3 минуты и помогла настроить API!')">💚 Благодарность (Positive)</button>
            <button type="button" class="sample-btn" onclick="setSample('С моей карты ошибочно списали $89 за годовой тариф, требую немедленный возврат денег!')">💳 Списание денег (Billing)</button>
            <button type="button" class="sample-btn" onclick="setSample('Application crashes with NullPointerException error 500 when exporting large CSV report')">🛠️ Ошибка в коде (Bug)</button>
            <button type="button" class="sample-btn" onclick="setSample('URGENT: Suspicious login from IP 194.26.29.1, password was changed without my permission!')">🚨 Взлом аккаунта (Security)</button>
        </div>

        <label for="text_input">Введите произвольный текст на русском или английском:</label>
        <textarea id="text_input" name="text_input" required placeholder="Введите текст...">{{ input_text if input_text else '' }}</textarea>
        <button type="submit" class="btn-run">⚡ Запустить трансформер и извлечь эмбеддинг</button>
    </form>

    {% if result %}
    <div class="result-box">
        <h3>Результаты инференса трансформера</h3>
        
        <div class="top-badge">
            Итоговая категория: {{ result.top_category }} (Уверенность: {{ result.confidence }}%)
        </div>

        <!-- Таблица вероятностей и сходства -->
        <table class="prob-table">
            <thead>
                <tr>
                    <th>Категория / Намерение</th>
                    <th style="width: 35%;">Вероятность модели</th>
                    <th>Косинусное сходство (Zero-Shot)</th>
                </tr>
            </thead>
            <tbody>
                {% for c in result.classes %}
                <tr>
                    <td><b>{{ c.emoji }} {{ c.label }}</b></td>
                    <td>
                        {{ c.prob }}%
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: {{ c.prob }}%;"></div>
                        </div>
                    </td>
                    <td><b>{{ c.similarity }}%</b></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <!-- Инспекция токенов -->
        <div style="margin-top: 25px;">
            <h4>🔍 Как трансформер «прочитал» текст (WordPiece Токенизация):</h4>
            <div style="font-size: 13px; color: #64748b; margin-bottom: 6px;">
                Всего токенов: <b>{{ result.token_info.num_tokens }}</b> (включая спец-токены [CLS] и [SEP])
            </div>
            <div class="token-pills">
                {% for tok in result.token_info.tokens %}
                <span class="token-pill {{ 'special' if tok in ['[CLS]', '[SEP]', '[PAD]', '[UNK]'] else '' }}">{{ tok }}</span>
                {% endfor %}
            </div>
        </div>

        <!-- Срез вектора эмбеддинга -->
        <div style="margin-top: 25px;">
            <h4>📐 Плотный вектор эмбеддинга (Dense Embedding, 384 измерения):</h4>
            <div style="font-size: 12px; color: #64748b;">Первые 10 вещественных компонент L2-нормализованного вектора:</div>
            <div class="vector-box">{{ result.vector_preview }}</div>
        </div>

        <!-- Matryoshka Truncation -->
        <div style="margin-top: 20px;">
            <h4>🪆 Matryoshka Representation Learning (MRL Срез до 64 dims):</h4>
            <div style="font-size: 12px; color: #64748b;">Усечение вектора в 6 раз с сохранением геометрического порядка:</div>
            <div class="vector-box" style="color: #4ade80;">{{ result.matryoshka_preview }}</div>
        </div>
    </div>
    {% endif %}

    <div class="info-panel">
        <b>💡 Почему трансформер превосходит TF-IDF?</b><br>
        1. <b>Self-Attention:</b> Каждое слово «смотрит» на все остальные слова во фразе, улавливая иронию, отрицания («не работает» vs «работает») и контекст.<br>
        2. <b>Mean Pooling:</b> Выходные векторы всех токенов усредняются с учетом маски внимания, образуя единую семантическую точку в 384-мерном пространстве.<br>
        3. <b>MRL (Matryoshka):</b> При хранении миллионов документов в векторной БД размерность можно динамически усекать до 64–128 чисел для экономии RAM без потери точности.
    </div>
</div>

<script>
function setSample(text) {
    document.getElementById('text_input').value = text;
}
</script>

</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def transformer_view():
    result = None
    input_text = ""

    if request.method == 'POST':
        input_text = request.form.get('text_input', '').strip()
        if input_text:
            result = pipeline.predict(input_text)

    return render_template_string(HTML_TEMPLATE, result=result, input_text=input_text)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Local Transformer NLP Pipeline запущен!")
    print("👉 Откройте в браузере: http://127.0.0.1:5007/")
    print("="*70 + "\n")
    app.run(debug=True, port=5007)
