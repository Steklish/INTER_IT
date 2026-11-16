"""
===============================================================================
FLASK USE CASE 4: АУДИТ ФИНАНСОВЫХ ОПЕРАЦИЙ И ПОИСК АНОМАЛИЙ (FILE UPLOAD)
===============================================================================

1. В ЧЕМ СУТЬ И ПОЧЕМУ ЭТО НЕ ТРИВИАЛЬНО?
----------------------------------------
Это реализация задачи BATCH-СКОРИНГА (пакетной обработки файлов) и обучения БЕЗ
учителя (Unsupervised Anomaly Detection) с помощью алгоритма Isolation Forest.

В антифрод-системах банков миллионы транзакций происходят ежеминутно. Размечать
каждую мошенническую операцию вручную невозможно.
Алгоритм Isolation Forest (Изолирующий лес):
  1. Строит случайные деревья решений.
  2. Нормальные транзакции (например, покупка кофе за $4 в 14:00) лежат плотно в центре
     облака данных, поэтому чтобы изолировать их, требуется много разбиений (глубокие ветви).
  3. Аномалии (например, перевод $48,000 в 03:40 ночи из зоны риска) лежат на периферии
     и изолируются буквально за 1–2 случайных сплита!
  4. Модель рассчитывает Anomaly Score и автоматически помечает подозрительные строки.

2. КАК ЭТО РАБОТАЕТ В FLASK?
----------------------------
  - Пользователь загружает файл `.csv` через форму `<input type="file">`.
  - Сервер считывает CSV прямо из оперативной памяти без лишней записи на диск.
  - Пайплайн прогоняет датасет через `IsolationForest.predict()` и `decision_function()`.
  - Страница отображает сводный дашборд и подсвечивает подозрительные транзакции красным!
  - Также встроена кнопка «📥 Скачать тестовый CSV», чтобы вам не пришлось создавать файл вручную.

3. КАК ЗАПУСТИТЬ:
-----------------
    python app_4_fraud_detector.py
Затем откройте в браузере: http://127.0.0.1:5004/
===============================================================================
"""

import os
import io
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template_string, Response, send_file

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "anomaly_detector_pipeline.joblib"
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_transactions.csv"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


# =============================================================================
# ШАГ 1: ГЕНЕРАЦИЯ ОБРАЗЦА CSV И ОБУЧЕНИЕ МОДЕЛИ ИЗОЛИРУЮЩЕГО ЛЕСА
# =============================================================================
def generate_sample_csv(file_path: Path):
    """Генерирует реалистичный журнал банковских транзакций с редкими аномалиями."""
    np.random.seed(42)
    n = 200

    # 1. 95% нормальных транзакций (покупки в магазинах, кафе, оплата ЖКУ)
    normal_amounts = np.random.exponential(scale=35.0, size=int(n * 0.95)) + 5.0
    normal_hours = np.random.normal(loc=14.0, scale=3.5, size=int(n * 0.95))
    normal_hours = np.clip(normal_hours, 7.0, 23.0)
    normal_velocity = np.random.choice([1, 2, 3], p=[0.75, 0.20, 0.05], size=int(n * 0.95))

    # 2. 5% аномальных транзакций (ночные переводы огромных сумм, всплески частоты)
    anom_amounts = np.random.uniform(5000, 45000, size=int(n * 0.05))
    anom_hours = np.random.choice([1, 2, 3, 4], size=int(n * 0.05))
    anom_velocity = np.random.randint(8, 25, size=int(n * 0.05))

    amounts = np.concatenate([normal_amounts, anom_amounts])
    hours = np.concatenate([normal_hours, anom_hours])
    velocity = np.concatenate([normal_velocity, anom_velocity])

    # Перемешиваем строки
    indices = np.arange(len(amounts))
    np.random.shuffle(indices)

    df = pd.DataFrame({
        'transaction_id': [f"TX-{10000 + i}" for i in range(len(amounts))],
        'amount_usd': np.round(amounts[indices], 2),
        'hour_of_day': np.round(hours[indices], 1),
        'velocity_1h': velocity[indices],
        'merchant': np.random.choice(['Supermarket', 'Coffee Shop', 'Electronics', 'Crypto Exchange', 'Online Game'], size=len(amounts))
    })

    df.to_csv(file_path, index=False)
    print(f"--> [Готово] Образец транзакций сохранен в {file_path}")
    return df


def train_and_save_anomaly_model():
    print("--> [Обучение] Обучение Isolation Forest для обнаружения аномалий...")
    if not SAMPLE_DATA_PATH.exists():
        df = generate_sample_csv(SAMPLE_DATA_PATH)
    else:
        df = pd.read_csv(SAMPLE_DATA_PATH)

    feature_cols = ['amount_usd', 'hour_of_day', 'velocity_1h']
    X = df[feature_cols]

    # Пайплайн: RobustScaler (устойчив к выбросам) + IsolationForest
    # contamination=0.05: ожидаем ~5% аномалий в потоке данных
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('isolation_forest', IsolationForest(
            n_estimators=150,
            contamination=0.05,
            random_state=42
        ))
    ])

    pipeline.fit(X)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"--> [Готово] Модель сохранена в: {MODEL_PATH}")
    return pipeline


if not SAMPLE_DATA_PATH.exists():
    generate_sample_csv(SAMPLE_DATA_PATH)

if not MODEL_PATH.exists():
    anomaly_model = train_and_save_anomaly_model()
else:
    print(f"--> Загружаем готовую модель из {MODEL_PATH}")
    anomaly_model = joblib.load(MODEL_PATH)


# =============================================================================
# ШАГ 2: HTML-ШАБЛОН СТРАНИЦЫ ЗАГРУЗКИ И ДАШБОРДА
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Fraud & Anomaly Audit</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 25px; }
        .container { max-width: 950px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }
        h1 { color: #0f172a; margin-top: 0; font-size: 24px; }
        .desc { color: #64748b; font-size: 14px; margin-bottom: 25px; }
        
        /* Зона загрузки файла */
        .upload-box { border: 2px dashed #cbd5e1; border-radius: 8px; padding: 25px; text-align: center; background: #f8fafc; margin-bottom: 20px; }
        .upload-box:hover { border-color: #3b82f6; background: #f0f7ff; }
        input[type="file"] { margin: 15px 0; font-size: 14px; }
        .btn-audit { background: #dc2626; color: #fff; border: none; padding: 12px 24px; font-size: 15px; font-weight: bold; border-radius: 6px; cursor: pointer; }
        .btn-audit:hover { background: #b91c1c; }
        .btn-download { background: #475569; color: #fff; text-decoration: none; padding: 8px 14px; border-radius: 6px; font-size: 13px; display: inline-block; margin-top: 10px; }
        .btn-download:hover { background: #334155; }
        
        /* Сводные метрики */
        .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 25px 0; }
        .summary-card { padding: 16px; border-radius: 8px; background: #f1f5f9; }
        .summary-label { font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; }
        .summary-val { font-size: 24px; font-weight: 800; color: #0f172a; margin-top: 4px; }
        
        /* Таблица */
        .table-responsive { max-height: 450px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 6px; margin-top: 15px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
        th { background: #f8fafc; position: sticky; top: 0; padding: 10px 14px; border-bottom: 2px solid #cbd5e1; color: #334155; }
        td { padding: 10px 14px; border-bottom: 1px solid #f1f5f9; }
        
        /* Подсветка подозрительных строк */
        .row-anomaly { background: #fef2f2 !important; color: #991b1b; font-weight: 600; }
        .badge-danger { background: #fee2e2; color: #dc2626; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .badge-ok { background: #dcfce7; color: #16a34a; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
    </style>
</head>
<body>

<div class="container">
    <h1>🛡️ Пакетный аудит транзакций и детектор фрода (Isolation Forest)</h1>
    <div class="desc">Пример Unsupervised Anomaly Detection: загрузка CSV-файла, поиск подозрительных всплесков активности в памяти.</div>

    <div class="upload-box">
        <form method="POST" enctype="multipart/form-data">
            <b>Загрузите файл реестра транзакций (.csv):</b><br>
            <input type="file" name="file" accept=".csv" required><br>
            <button type="submit" class="btn-audit">🔍 Запустить ML-аудит реестра</button>
        </form>
        <div style="margin-top: 15px;">
            <span style="font-size: 13px; color: #64748b;">Нет готового файла?</span><br>
            <a href="/download_sample" class="btn-download">📥 Скачать готовый тестовый sample_transactions.csv</a>
        </div>
    </div>

    {% if error %}
        <div style="background: #fee2e2; border: 1px solid #fca5a5; color: #991b1b; padding: 14px; border-radius: 6px;">
            <b>Ошибка:</b> {{ error }}
        </div>
    {% endif %}

    {% if summary %}
    <div>
        <h3>Результаты аудита файла:</h3>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-label">Всего транзакций</div>
                <div class="summary-val">{{ summary.total }}</div>
            </div>
            <div class="summary-card" style="background: #f0fdf4;">
                <div class="summary-label">Штатные операции</div>
                <div class="summary-val" style="color: #16a34a;">{{ summary.normal }}</div>
            </div>
            <div class="summary-card" style="background: #fef2f2;">
                <div class="summary-label">Выявлено аномалий</div>
                <div class="summary-val" style="color: #dc2626;">{{ summary.anomalies }}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Доля риска</div>
                <div class="summary-val">{{ summary.anomaly_rate }}%</div>
            </div>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4>Таблица операций (подозрительные подсвечены красным):</h4>
            <span style="font-size: 12px; color: #64748b;">Отображаются первые 100 строк</span>
        </div>

        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>ID операции</th>
                        <th>Сумма ($)</th>
                        <th>Время суток (ч)</th>
                        <th>Частота (в час)</th>
                        <th>Продавец</th>
                        <th>Статус системы</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in rows %}
                    <tr class="{{ 'row-anomaly' if row.is_anomaly else '' }}">
                        <td><code>{{ row.transaction_id }}</code></td>
                        <td>${{ "{:,.2f}".format(row.amount_usd) }}</td>
                        <td>{{ row.hour_of_day }}</td>
                        <td>{{ row.velocity_1h }}</td>
                        <td>{{ row.merchant }}</td>
                        <td>
                            {% if row.is_anomaly %}
                                <span class="badge-danger">⚠️ ПОДОЗРИТЕЛЬНО (ФРОД)</span>
                            {% else %}
                                <span class="badge-ok">✓ В НОРМЕ</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""


# =============================================================================
# ШАГ 3: ОБРАБОТЧИКИ МАРШРУТОВ (UPLOAD & DOWNLOAD)
# =============================================================================
@app.route('/download_sample')
def download_sample():
    """Позволяет скачать сгенерированный тестовый CSV-файл прямо из браузера."""
    if not SAMPLE_DATA_PATH.exists():
        generate_sample_csv(SAMPLE_DATA_PATH)
    return send_file(SAMPLE_DATA_PATH, as_attachment=True, download_name="sample_transactions.csv")


@app.route('/', methods=['GET', 'POST'])
def fraud_audit():
    summary = None
    rows = []
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "Файл не был прикреплен к запросу."
            return render_template_string(HTML_TEMPLATE, error=error)

        file = request.files['file']
        if file.filename == '':
            error = "Файл не выбран."
            return render_template_string(HTML_TEMPLATE, error=error)

        try:
            # Читаем CSV прямо из входящего файлового потока оперативной памяти
            df = pd.read_csv(file)

            # Проверяем наличие необходимых признаков
            required_cols = ['amount_usd', 'hour_of_day', 'velocity_1h']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"В файле отсутствует обязательная колонка: '{col}'")

            # Выполняем инференс Isolation Forest
            # IsolationForest выдает:
            #   1 = нормальное наблюдение (Inlier)
            #  -1 = аномалия / выброс (Outlier)
            predictions = anomaly_model.predict(df[required_cols])
            df['is_anomaly'] = (predictions == -1)

            # Если в файле нет id или merchant, создадим их для красивого отображения
            if 'transaction_id' not in df.columns:
                df['transaction_id'] = [f"TX-{i+1}" for i in range(len(df))]
            if 'merchant' not in df.columns:
                df['merchant'] = 'Unknown'

            total = len(df)
            anomalies = int(df['is_anomaly'].sum())
            normal = total - anomalies
            anomaly_rate = round((anomalies / total) * 100, 1) if total > 0 else 0.0

            summary = {
                'total': total,
                'anomalies': anomalies,
                'normal': normal,
                'anomaly_rate': anomaly_rate
            }

            # Сортируем так, чтобы подозрительные транзакции шли в самом начале таблицы
            df_sorted = df.sort_values(by='is_anomaly', ascending=False)
            rows = df_sorted.head(100).to_dict(orient='records')

        except Exception as e:
            error = f"Ошибка при анализе файла: {str(e)}"

    return render_template_string(HTML_TEMPLATE, summary=summary, rows=rows, error=error)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Fraud & Anomaly Audit запущен!")
    print("👉 Откройте в браузере: http://127.0.0.1:5004/")
    print("="*70 + "\n")
    app.run(debug=True, port=5004)
