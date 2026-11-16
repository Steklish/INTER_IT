"""
===============================================================================
FLASK USE CASE 6: ИНТЕЛЛЕКТУАЛЬНЫЙ OCR-ПАЙПЛАЙН И ИЗВЛЕЧЕНИЕ СУЩНОСТЕЙ
===============================================================================

1. В ЧЕМ СУТЬ И ПОЧЕМУ ЭТО ВАЖНО В ML & COMPUTER VISION?
--------------------------------------------------------
Оптическое распознавание символов (OCR) лежит в основе автоматизации документооборота
(IDP: Intelligent Document Processing), обработки счетов, чеков и банковских выписок.

Здесь реализован сквозной пайплайн из 4 фаз:
  1. Компьютерное зрение (OpenCV):
     - Grayscale, шумоподавление Bilateral Filter.
     - Адаптивная бинаризация (Adaptive Gaussian / Otsu).
  2. Детекция текста (Localization):
     - Извлечение координат (Bounding Boxes) для каждого отдельного слова.
  3. Распознавание символов (Tesseract LSTM Engine):
     - Чтение текста на английском и русском языках с расчетом Confidence Score.
  4. Постобработка и извлечение сущностей (NLP & RegEx):
     - Автоматический парсинг номера инвойса, даты, итоговой суммы, email и телефона.

2. КАК ЭТО РАБОТАЕТ В FLASK?
----------------------------
Пользователь загружает фото документа или выбирает тестовый образец.
Сервер прогоняет картинку через OpenCV, выполняет OCR-инференс, рисует зеленые рамки
вокруг каждого слова, парсит ключевые бизнес-поля и выводит результат на экран.

3. КАК ЗАПУСТИТЬ:
-----------------
    python app_6_ocr_pipeline.py
Затем откройте в браузере: http://127.0.0.1:5006/
===============================================================================
"""

import os
import io
import base64
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from flask import Flask, request, render_template_string

# Импортируем наш OCR модуль
from ocr_pipeline import OCRPipeline, generate_sample_invoice_image

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# Инициализируем пайплайн
ocr_engine = OCRPipeline(lang="eng+rus")

# Подготовим превью тестовых образцов для мгновенного теста
SAMPLE_DATA_URIS = {}
for style_name in ["invoice", "bilingual_receipt"]:
    pil_img = generate_sample_invoice_image(style=style_name)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    SAMPLE_DATA_URIS[style_name] = f"data:image/jpeg;base64,{b64}"


def cv2_to_base64_data_uri(cv2_bgr_or_gray: np.ndarray, format_str: str = ".jpg") -> str:
    """Конвертирует OpenCV-изображение в Data URI для прямого отображения в HTML."""
    success, encoded_img = cv2.imencode(format_str, cv2_bgr_or_gray)
    if not success:
        return ""
    b64 = base64.b64encode(encoded_img).decode("utf-8")
    mime = "image/png" if format_str == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# =============================================================================
# HTML-ШАБЛОН СТРАНИЦЫ
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Intelligent OCR Pipeline</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 25px; }
        .container { max-width: 1050px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }
        h1 { color: #0f172a; margin-top: 0; font-size: 24px; }
        .desc { color: #64748b; font-size: 14px; margin-bottom: 25px; line-height: 1.5; }
        
        .upload-card { border: 2px dashed #cbd5e1; border-radius: 8px; padding: 25px; text-align: center; background: #f8fafc; }
        .upload-card:hover { border-color: #3b82f6; background: #f0f7ff; }
        .btn-ocr { background: #0284c7; color: #fff; border: none; padding: 12px 26px; font-size: 15px; font-weight: bold; border-radius: 6px; cursor: pointer; margin-top: 15px; }
        .btn-ocr:hover { background: #0369a1; }
        
        .controls-row { display: flex; justify-content: center; gap: 20px; margin-top: 12px; font-size: 13px; color: #334155; }
        select { padding: 6px 12px; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 13px; }
        
        .quick-samples { margin-top: 20px; text-align: left; }
        .samples-grid { display: flex; gap: 15px; margin-top: 8px; }
        .sample-item { border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; cursor: pointer; text-align: center; background: #fff; flex: 1; transition: transform 0.15s, border-color 0.15s; }
        .sample-item:hover { transform: translateY(-2px); border-color: #0284c7; box-shadow: 0 4px 10px rgba(2,132,199,0.15); }
        .sample-item img { width: 100%; height: 90px; object-fit: cover; border-radius: 4px; margin-bottom: 5px; }
        
        /* Сетка результатов */
        .results-section { margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 25px; }
        .metrics-bar { display: flex; gap: 15px; margin-bottom: 20px; }
        .metric-card { background: #f1f5f9; padding: 14px 20px; border-radius: 8px; flex: 1; }
        .metric-label { font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; }
        .metric-val { font-size: 22px; font-weight: 800; color: #0f172a; margin-top: 4px; }
        
        .entities-card { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 20px; margin-bottom: 25px; }
        .entities-title { font-size: 16px; font-weight: bold; color: #166534; margin-bottom: 12px; }
        .entities-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }
        .entity-item { background: #fff; padding: 10px 14px; border-radius: 6px; border: 1px solid #dcfce7; }
        .entity-key { font-size: 11px; color: #64748b; text-transform: uppercase; }
        .entity-val { font-size: 15px; font-weight: bold; color: #15803d; margin-top: 2px; }
        
        /* Сравнение изображений */
        .images-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }
        .img-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; }
        .img-box img { max-width: 100%; height: auto; max-height: 380px; border-radius: 4px; border: 1px solid #cbd5e1; }
        .img-caption { font-size: 12px; font-weight: 600; color: #475569; margin-bottom: 8px; }
        
        .fulltext-box { background: #0f172a; color: #e2e8f0; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 13px; white-space: pre-wrap; max-height: 250px; overflow-y: auto; }
        .info-card { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; padding: 16px; border-radius: 8px; font-size: 13px; line-height: 1.6; margin-top: 30px; }
    </style>
</head>
<body>

<div class="container">
    <h1>📄 Интеллектуальный OCR-пайплайн и извлечение данных (OpenCV + Tesseract)</h1>
    <div class="desc">
        Сквозной процесс обработки документов: адаптивная бинаризация изображений, локализация слов с Bounding Boxes, 
        LSTM-распознавание текста и автоматическое извлечение полей (номера инвойса, даты, суммы) регулярными выражениями.
    </div>

    <div class="upload-card">
        <form method="POST" enctype="multipart/form-data">
            <input type="hidden" name="sample_style" id="sample_style" value="">

            <b>Загрузите скан или фотографию документа (JPG / PNG / WEBP):</b><br>
            <input type="file" name="doc_image" accept="image/*"><br>

            <div class="controls-row">
                <div>
                    <label><b>Метод бинаризации:</b></label>
                    <select name="prep_method">
                        <option value="otsu" {{ 'selected' if prep_method == 'otsu' else '' }}>Метод Оцу (Otsu Thresholding - для сканов)</option>
                        <option value="adaptive" {{ 'selected' if prep_method == 'adaptive' else '' }}>Адаптивный порог (Adaptive - для фото с тенями)</option>
                        <option value="grayscale" {{ 'selected' if prep_method == 'grayscale' else '' }}>Только градация серого (Grayscale Denoise)</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="btn-ocr">🔍 Запустить сквозной OCR-анализ</button>
        </form>

        <div class="quick-samples">
            <span style="font-size: 13px; font-weight: 600; color: #475569;">Или протестируйте на готовых образцах:</span>
            <div class="samples-grid">
                <div class="sample-item" onclick="submitSample('invoice')">
                    <img src="{{ sample_uris['invoice'] }}" alt="Invoice">
                    <span style="font-size: 12px; font-weight: 600;">🧾 Официальный инвойс (INV-94821)</span>
                </div>
                <div class="sample-item" onclick="submitSample('bilingual_receipt')">
                    <img src="{{ sample_uris['bilingual_receipt'] }}" alt="Receipt">
                    <span style="font-size: 12px; font-weight: 600;">🛒 Чек супермаркета (REC-77412)</span>
                </div>
            </div>
        </div>
    </div>

    {% if error %}
        <div style="background: #fee2e2; border: 1px solid #fca5a5; color: #991b1b; padding: 14px; border-radius: 6px; margin-top: 20px;">
            <b>Ошибка:</b> {{ error }}
        </div>
    {% endif %}

    {% if result %}
    <div class="results-section">
        <h3>Результаты работы пайплайна</h3>

        <!-- Сводные метрики качества OCR -->
        <div class="metrics-bar">
            <div class="metric-card">
                <div class="metric-label">Всего слов найдено</div>
                <div class="metric-val">{{ result.total_words }}</div>
            </div>
            <div class="metric-card" style="background: #f0fdf4;">
                <div class="metric-label">Средняя уверенность (Confidence)</div>
                <div class="metric-val" style="color: #16a34a;">{{ result.avg_confidence }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Метод предобработки</div>
                <div class="metric-val" style="font-size: 18px;">{{ result.prep_method_label }}</div>
            </div>
        </div>

        <!-- Извлеченные сущности -->
        <div class="entities-card">
            <div class="entities-title">🎯 Автоматически извлеченные ключевые поля (Structured Entities):</div>
            <div class="entities-grid">
                <div class="entity-item">
                    <div class="entity-key">Номер документа</div>
                    <div class="entity-val">{{ result.entities.document_id }}</div>
                </div>
                <div class="entity-item">
                    <div class="entity-key">Дата операции</div>
                    <div class="entity-val">{{ result.entities.date }}</div>
                </div>
                <div class="entity-item">
                    <div class="entity-key">Итоговая сумма</div>
                    <div class="entity-val" style="color: #dc2626;">{{ result.entities.total_amount }}</div>
                </div>
                <div class="entity-item">
                    <div class="entity-key">Email плательщика</div>
                    <div class="entity-val" style="font-size: 13px;">{{ result.entities.email }}</div>
                </div>
                <div class="entity-item">
                    <div class="entity-key">Телефон</div>
                    <div class="entity-val">{{ result.entities.phone }}</div>
                </div>
            </div>
        </div>

        <!-- Визуализация Bounding Boxes и предобработки -->
        <div class="images-grid">
            <div class="img-box">
                <div class="img-caption">1. Бинаризация OpenCV (Очистка от теней и шума):</div>
                <img src="{{ result.preprocessed_uri }}" alt="Preprocessed Binary">
            </div>
            <div class="img-box">
                <div class="img-caption">2. Детекция слов и Bounding Boxes (Зеленые рамки):</div>
                <img src="{{ result.annotated_uri }}" alt="Annotated Words">
            </div>
        </div>

        <!-- Полный извлеченный текст -->
        <h4>Полный распознанный текст документа:</h4>
        <div class="fulltext-box">{{ result.full_text }}</div>
    </div>
    {% endif %}

    <div class="info-card">
        <b>💡 Почему качественная предобработка в OCR решает всё?</b><br>
        Если скормить нейросети Tesseract сырую фотографию с камеры телефона, неравномерное освещение, 
        тени от пальцев и сероватый фон приводят к галлюцинациям (символы двоятся или теряются).<br>
        1. <b>Bilateral Filter</b> удаляет зернистость матрицы камеры, но оставляет ребра букв идеально резкими.<br>
        2. <b>Бинаризация (Otsu / Adaptive)</b> математически разделяет пиксели чернил и фона, гарантируя чистые контуры.<br>
        3. <b>Regular Expressions (RegEx)</b> на финальном этапе превращают массив распознанных строк в готовую запись для базы данных.
    </div>
</div>

<script>
function submitSample(style) {
    document.getElementById('sample_style').value = style;
    document.forms[0].submit();
}
</script>

</body>
</html>
"""


# =============================================================================
# ОБРАБОТЧИК ЗАПРОСОВ FLASK
# =============================================================================
@app.route('/', methods=['GET', 'POST'])
def ocr_view():
    result = None
    error = None
    prep_method = "otsu"

    if request.method == 'POST':
        prep_method = request.form.get('prep_method', 'otsu')
        bgr_image = None

        # Проверяем клик по быстрому образцу
        sample_style = request.form.get('sample_style', '')
        if sample_style in ['invoice', 'bilingual_receipt']:
            pil_sample = generate_sample_invoice_image(style=sample_style)
            bgr_image = np.array(pil_sample)[:, :, ::-1]

        # Иначе проверяем загруженный пользователем файл
        if bgr_image is None and 'doc_image' in request.files:
            file = request.files['doc_image']
            if file and file.filename != '':
                try:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if bgr_image is None:
                        raise ValueError("Не удалось декодировать изображение.")
                except Exception as e:
                    error = f"Ошибка чтения файла: {str(e)}"

        if bgr_image is not None and error is None:
            try:
                # 1. Предобработка OpenCV
                gray, preprocessed = ocr_engine.preprocess_image(bgr_image, method=prep_method)

                # 2. Детекция и распознавание текста
                ocr_data = ocr_engine.run_ocr(preprocessed)

                # 3. Отрисовка Bounding Boxes на исходном цветном изображении
                annotated_bgr = ocr_engine.draw_annotated_boxes(bgr_image, ocr_data['words'])

                # 4. Извлечение структурированных сущностей
                entities = ocr_engine.extract_structured_entities(ocr_data['full_text'])

                # Конвертируем изображения в base64 для показа в браузере
                preprocessed_uri = cv2_to_base64_data_uri(preprocessed, format_str=".png")
                annotated_uri = cv2_to_base64_data_uri(annotated_bgr, format_str=".jpg")

                method_labels = {
                    'otsu': 'Otsu Thresholding (Оцу)',
                    'adaptive': 'Adaptive Gaussian (Адаптивный)',
                    'grayscale': 'Grayscale Denoise (Серый)'
                }

                result = {
                    'full_text': ocr_data['full_text'],
                    'total_words': ocr_data['total_words'],
                    'avg_confidence': ocr_data['avg_confidence'],
                    'prep_method_label': method_labels.get(prep_method, prep_method),
                    'entities': entities,
                    'preprocessed_uri': preprocessed_uri,
                    'annotated_uri': annotated_uri
                }

            except Exception as e:
                error = f"Ошибка выполнения OCR-пайплайна: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        error=error,
        prep_method=prep_method,
        sample_uris=SAMPLE_DATA_URIS
    )


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Intelligent OCR Pipeline запущен!")
    print("👉 Откройте в браузере: http://127.0.0.1:5006/")
    print("="*70 + "\n")
    app.run(debug=True, port=5006)
