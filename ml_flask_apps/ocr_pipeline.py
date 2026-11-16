"""
===============================================================================
MODULE: OCR PIPELINE (Оптическое распознавание символов)
===============================================================================

1. ЧТО ТАКОЕ OCR И ИЗ КАКИХ ЭТАПОВ СОСТОИТ НАСТОЯЩИЙ ПРОМЫШЛЕННЫЙ ПАЙПЛАЙН?
--------------------------------------------------------------------------
Большинство новичков думают, что OCR — это просто вызов `tesseract(image)`.
Но в 95% реальных бизнес-задач (чеки, накладные, сканы паспортов, фотографии на телефон)
«сырое» изображение дает мусор: шум, тени, перекосы и блики убивают точность.

Настоящий промышленный OCR-пайплайн состоит из 4 последовательных этапов:

  [1. Image Preprocessing (Компьютерное зрение / OpenCV)]
        - Grayscale (перевод в градации серого)
        - Bilateral Filtering / Gaussian Blur (шумоподавление с сохранением резкости букв)
        - Adaptive Gaussian Thresholding / Otsu (бинаризация: разделение чернил и бумаги)
        - Deskewing (компенсация угла поворота и выравнивание строк)
        - Morphological Closing (закрытие микроразрывов в символах)
        │
        ▼
  [2. Text Detection & Localization (Поиск областей текста)]
        - Поиск координат каждого слова (Bounding Boxes: x, y, width, height)
        │
        ▼
  [3. Text Recognition & Sequence Decoding (Распознавание символов)]
        - Рекуррентная сеть LSTM / CRNN с CTC-Loss (Connectionist Temporal Classification)
        - Расчет вероятности (Confidence score) для каждого распознанного слова
        │
        ▼
  [4. Post-Processing & Entity Extraction (Извлечение сущностей / RegEx & NLP)]
        - Очистка опечаток OCR (например, путаницы l/1/I или O/0)
        - Автоматическое извлечение ключевых полей документа:
          * Номер накладной/чека (Invoice #)
          * Дата операции (DD.MM.YYYY)
          * Итоговая сумма к оплате ($ / руб)
          * Email и контактные данные

===============================================================================
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pytesseract

# -----------------------------------------------------------------------------
# НАСТРОЙКА ПУТЕЙ К ДВИЖКУ TESSERACT
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TESSDATA_DIR = BASE_DIR / "tessdata"
TESSDATA_DIR.mkdir(parents=True, exist_ok=True)

# Задаем путь к исполняемому файлу Tesseract в системе Windows
POSSIBLE_PATHS = [
    r"C:\msys64\ucrt64\bin\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break

os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


class OCRPipeline:
    """
    Промышленный сквозной пайплайн OCR:
    OpenCV Предобработка -> Tesseract LSTM Детекция -> Bounding Boxes -> Извлечение сущностей.
    """

    def __init__(self, lang: str = "eng+rus"):
        self.lang = lang
        # Передаем путь с прямыми слэшами и без лишних кавычек для кросс-платформенности
        tess_clean_path = str(TESSDATA_DIR).replace('\\', '/')
        self.tess_config = f'--tessdata-dir {tess_clean_path} --oem 3 --psm 3'

    def preprocess_image(self, bgr_image: np.ndarray, method: str = "otsu") -> Tuple[np.ndarray, np.ndarray]:
        """
        ЭТАП 1: Компьютерное зрение (OpenCV).
        Превращает сырое фото с шумом и неравномерным светом в чистое черно-белое изображение.
        """
        # 1. Перевод в градации серого
        if len(bgr_image.shape) == 3:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_image.copy()

        # 2. Фильтрация шумов (Bilateral Filter сохраняет четкие границы шрифта)
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # 3. Бинаризация (Thresholding)
        if method == "adaptive":
            # Адаптивный порог: идеален для фото чеков с тенями от рук или неравномерной вспышкой
            binary = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15, C=4
            )
        elif method == "otsu":
            # Метод Оцу: автоматически находит оптимальный порог контраста
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Метод grayscale / denoise_only
            binary = denoised

        return gray, binary

    def run_ocr(self, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        """
        ЭТАП 2 и 3: Детекция рамок слов и распознавание текста нейросетью.
        """
        # Получаем детальные данные по каждому слову (координаты + уверенность)
        data = pytesseract.image_to_data(
            preprocessed_image,
            lang=self.lang,
            config=self.tess_config,
            output_type=pytesseract.Output.DICT
        )

        words_info = []
        n_boxes = len(data['text'])
        full_text_tokens = []
        confidences = []

        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])

            # Tesseract выдает conf=-1 для блоков разметки без слов
            if text and conf > 0:
                words_info.append({
                    'text': text,
                    'conf': round(conf, 1),
                    'box': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                })
                full_text_tokens.append(text)
                confidences.append(conf)

        full_text = pytesseract.image_to_string(
            preprocessed_image,
            lang=self.lang,
            config=self.tess_config
        ).strip()

        avg_conf = round(float(np.mean(confidences)), 1) if confidences else 0.0

        return {
            'full_text': full_text,
            'words': words_info,
            'avg_confidence': avg_conf,
            'total_words': len(words_info)
        }

    def draw_annotated_boxes(self, original_bgr: np.ndarray, words_info: List[Dict[str, Any]]) -> np.ndarray:
        """
        Рисует Bounding Boxes (зеленые и синие рамки) вокруг каждого распознанного слова
        с индикатором уверенности распознавания.
        """
        annotated = original_bgr.copy()
        for item in words_info:
            x, y, w, h = item['box']
            conf = item['conf']

            # Зеленый = высокая уверенность (>80%), Оранжевый = средняя, Красный = низкая (<50%)
            if conf >= 80:
                color = (0, 200, 0)
            elif conf >= 50:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        return annotated

    def extract_structured_entities(self, text: str) -> Dict[str, Any]:
        """
        ЭТАП 4: Постобработка и парсинг сущностей документа с помощью RegEx.
        Автоматически находит номер документа, суммы, даты, email и телефоны.
        """
        entities = {}

        # 1. Поиск номера накладной / счета / чека (требует знак :, #, № или дефис)
        invoice_match = re.search(r'(?:invoice|чек|счет|order|inv|№|#)\s*[:#&№-]+\s*([A-Za-z0-9\-_]{3,20})', text, re.IGNORECASE)
        entities['document_id'] = invoice_match.group(1) if invoice_match else "Не обнаружен"

        # 2. Поиск итоговой суммы
        total_match = re.search(r'(?:total|итог|итого|amount|sum|к\s*оплате|всего)\s*[:=]?\s*[\$€₽,]?\s*([\d\s,]+[\.\,]\d{2})', text, re.IGNORECASE)
        if total_match:
            clean_sum = total_match.group(1).replace(" ", "").replace(",", ".")
            entities['total_amount'] = f"${clean_sum}"
        else:
            generic_sum = re.search(r'[\$€₽]\s*([\d\s,]+[\.\,]\d{2})', text)
            entities['total_amount'] = generic_sum.group(0) if generic_sum else "Не обнаружено"

        # 3. Поиск дат (DD.MM.YYYY, YYYY-MM-DD, etc.)
        date_match = re.search(r'\b(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4}|\d{4}[\.\/\-]\d{1,2}[\.\/\-]\d{1,2})\b', text)
        entities['date'] = date_match.group(1) if date_match else "Не обнаружена"

        # 4. Поиск Email (с поддержкой возможных тире/дефисов OCR)
        clean_text_for_email = text.replace('—', '-').replace('–', '-')
        email_match = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', clean_text_for_email)
        entities['email'] = email_match.group(0) if email_match else "Не указан"

        # 5. Поиск телефонов
        phone_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}', text)
        entities['phone'] = phone_match.group(0) if phone_match else "Не указан"

        return entities


# =============================================================================
# ГЕНЕРАТОР ТЕСТОВЫХ ОБРАЗЦОВ ДОКУМЕНТОВ ДЛЯ ДЕМОНСТРАЦИИ
# =============================================================================
def generate_sample_invoice_image(style: str = "invoice") -> Image.Image:
    """Создает реалистичные четкие тестовые документы с помощью OpenCV и Pillow."""
    if style == "invoice":
        img = np.full((550, 850, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (850, 70), (45, 30, 20), -1)
        cv2.putText(img, 'TECHNO-SOLUTIONS // OFFICIAL INVOICE', (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, 'INVOICE: INV-94821', (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, 'DATE: 14.11.2026', (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, 'CLIENT: Alexander Petrov', (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, 'EMAIL: billing@techno-solutions.com', (40, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(img, (40, 300), (800, 300), (200, 200, 200), 2)
        cv2.putText(img, '1. Cloud Server Enterprise    $850.00', (40, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(img, '2. SSL Certificate 2-Year     $120.00', (40, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(img, '3. Technical Support Setup    $480.00', (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.line(img, (40, 450), (800, 450), (200, 200, 200), 2)
        cv2.putText(img, 'TOTAL: $1,450.00', (480, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 3)

        return Image.fromarray(img)

    elif style == "bilingual_receipt":
        img = np.full((550, 750, 3), 250, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (750, 60), (30, 80, 30), -1)
        cv2.putText(img, 'GLOBUS SUPERMARKET // RECEIPT', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(img, 'ORDER: REC-77412', (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, 'DATE: 16.11.2026', (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, 'PHONE: +7 (495) 123-45-67', (40, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(img, (40, 240), (700, 240), (180, 180, 180), 2)
        cv2.putText(img, 'Coffee Arabica Beans 1kg     $25.00', (40, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(img, 'Organic Honey & Oatmeal       $8.50', (40, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(img, 'Ceylon Black Tea 100pk        $6.50', (40, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.line(img, (40, 410), (700, 410), (180, 180, 180), 2)
        cv2.putText(img, 'TOTAL: $40.00', (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 120, 20), 3)
        cv2.putText(img, 'PAID VIA MIR CARD *4819', (40, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 2)

        return Image.fromarray(img)
