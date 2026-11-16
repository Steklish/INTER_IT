"""
===============================================================================
MODULE: TRANSFORMER PIPELINE (Эмбеддинги и классификация через Трансформеры)
===============================================================================

1. ЧТО ТАКОЕ ТРАНСФОРМЕР И КАК ОН ИЗВЛЕКАЕТ ПРИЗНАКИ?
---------------------------------------------------
Трансформеры (архитектура на базе механизма Self-Attention) произвели революцию
в обработке естественного языка (NLP) и мультимодальных данных.

В отличие от старых методов (TF-IDF или Bag-of-Words), которые просто считают слова:
  1. Трансформер понимает КОНТЕКСТ и МНОГОЗНАЧНОСТЬ:
     Слово «ключ» в фразах «гаечный ключ», «ключ от замка» и «скрипичный ключ»
     получит совершенно разные векторные представления!
  2. Трансформер сжимает весь смысл предложения в плотный вектор (Dense Embedding)
     фиксированной длины (обычно 384, 768 или 1024 числа).

2. ЭТАПЫ ИЗВЛЕЧЕНИЯ ЭМБЕДДИНГА ИЗ ТЕКСТА:
-----------------------------------------
  [ Сырой текст: "Отличный сервис, заказ приехал вовремя!" ]
              │
              ▼ (Шаг 1: Токенизация через AutoTokenizer)
  [ Токены (WordPieces) + Special Tokens ([CLS], [SEP]) + Attention Mask ]
              │
              ▼ (Шаг 2: Прямой проход через трансформер AutoModel)
  [ Тензор скрытых состояний: Shape (Batch, Seq_Len, Hidden_Dim=384) ]
              │
              ▼ (Шаг 3: Пулинг / Mean Pooling с учетом маски внимания)
  [ Единый вектор предложения: Shape (384,) ]
              │
              ▼ (Шаг 4: L2-Нормализация)
  [ Нормализованный эмбеддинг на единичной сфере ||v|| = 1.0 ]

3. ДВА СПОСОБА КЛАССИФИКАЦИИ НА ТРАНСФОРМЕРАХ:
---------------------------------------------
  - Способ А (Zero-Shot / Metric Learning):
    Сравниваем эмбеддинг входного текста с эмбеддингами эталонных описаний классов
    через косинусное сходство (Cosine Similarity). Не требует обучения!
  - Способ Б (Downstream Head):
    Подаем эмбеддинги в легковесную модель (LogisticRegression, LightGBM или MLP).
    Обучается за доли секунды и дает высокую точность.
===============================================================================
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_SAVE_PATH = BASE_DIR / "models" / "transformer_classifier.joblib"
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Имя компактного трансформера (~80 МБ, 384 размерность)
TRANSFORMER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TransformerFeatureExtractor:
    """
    Легковесный экстрактор эмбеддингов на базе Hugging Face Transformers.
    Работает локально на CPU без внешних сетевых вызовов (после кэширования весов).
    """

    def __init__(self, model_name: str = TRANSFORMER_NAME, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Режим инференса

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Превращает один текст или список текстов в матрицу эмбеддингов (N, 384).
        Реализует Mean Pooling с учетом Attention Mask.
        """
        if isinstance(texts, str):
            texts = [texts]

        # 1. Токенизация с дополнением (padding) и обрезкой (truncation) до 256 токенов
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_inputs)
            last_hidden_state = model_output.last_hidden_state  # [Batch, Seq_Len, Hidden_Dim]

            # 2. Mean Pooling: усредняем эмбеддинги только реальных токенов (игнорируя padding)
            attention_mask = encoded_inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            # 3. L2-нормализация (скалярное произведение равно косинусному расстоянию)
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def inspect_tokenization(self, text: str) -> Dict[str, Any]:
        """Возвращает наглядную структуру того, как трансформер видит текст изнутри."""
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        return {
            "tokens": tokens,
            "token_ids": input_ids,
            "num_tokens": len(tokens)
        }


# =============================================================================
# ОБУЧАЮЩИЙ И ИНФЕРЕНС-КЛАССИФИКАТОР НА ЭМБЕДДИНГАХ
# =============================================================================
class TransformerClassifierPipeline:
    """
    Сквозной гибрид: Transformer Feature Extractor + Linear Classifier Head.
    Решает задачу анализа тональности и намерений клиентов:
      0: Положительный отзыв / Благодарность (Positive / Delight)
      1: Жалоба / Проблема с оплатой (Negative / Billing Issue)
      2: Технический вопрос / Баг (Technical Query / Bug)
      3: Срочный инцидент безопасности (Security Alert)
    """

    CATEGORIES = [
        "Положительный отзыв / Благодарность (Positive)",
        "Жалоба на оплату или подписку (Billing Dispute)",
        "Техническая консультация / Баг (Technical Support)",
        "Критический инцидент безопасности (Security Alert)"
    ]

    CATEGORY_EMOJIS = ["💚", "💳", "🛠️", "🚨"]

    # Эталонные прототипы для Zero-Shot ранжирования
    PROTOTYPE_TEXTS = [
        "Great service, everything works perfectly, thank you very much for great customer experience!",
        "Money was charged incorrectly, wrong invoice, double payment, refund required immediately.",
        "Error 500, application crashes, bug in code, API endpoint failure and connection timeout.",
        "Account hacked, unauthorized login, credentials leaked, suspicious access from unknown IP."
    ]

    def __init__(self):
        self.extractor = TransformerFeatureExtractor()
        self.classifier = None
        self.prototype_embeddings = self.extractor.encode(self.PROTOTYPE_TEXTS, normalize=True)

        if MODEL_SAVE_PATH.exists():
            self.classifier = joblib.load(MODEL_SAVE_PATH)
        else:
            self._train_default_classifier()

    def _train_default_classifier(self):
        """Обучает классификационную голову на компактной выборке интентов."""
        training_samples = [
            # 0: Positive
            ("Awesome platform, saves me so much time every day!", 0),
            ("Большое спасибо службе поддержки, решили вопрос за 5 минут, всё супер!", 0),
            ("Amazing product quality and very intuitive user interface.", 0),
            ("Очень доволен сервисом, рекомендую всем коллегам!", 0),

            # 1: Billing
            ("Why did you charge my credit card 50 dollars without authorization?", 1),
            ("Списали деньги за отмененную подписку, требую немедленного возврата средств!", 1),
            ("Invoice number is incorrect and tax calculation is wrong.", 1),
            ("Не могу оплатить счет, карта отклонена банком, разберитесь с чеком.", 1),

            # 2: Technical
            ("Application throws null pointer exception when exporting CSV data.", 2),
            ("После последнего обновления кнопка в личном кабинете перестала нажиматься.", 2),
            ("Server returns 502 Bad Gateway error on API POST request.", 2),
            ("Вылетает мобильное приложение при переходе в настройки профиля.", 2),

            # 3: Security
            ("Unauthorized login attempt detected from foreign IP address, please lock account!", 3),
            ("Кажется меня взломали, пароль сброшен и включена чужая двухфакторка 2FA!", 3),
            ("Security breach alert, database credentials exposed publicly on Github.", 3),
            ("Подозрительная активность в профиле, кто-то изменил привязанную почту.", 3)
        ]

        texts = [s[0] for s in training_samples]
        labels = [s[1] for s in training_samples]

        embeddings = self.extractor.encode(texts, normalize=True)
        clf = LogisticRegression(C=2.0, max_iter=200, random_state=42)
        clf.fit(embeddings, labels)

        self.classifier = clf
        joblib.dump(clf, MODEL_SAVE_PATH)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Выполняет инференс:
        1. Извлекает нормализованный эмбеддинг текста (384-мерный вектор).
        2. Рассчитывает косинусное сходство с эталонными прототипами (Zero-Shot).
        3. Вычисляет калиброванные вероятности через логистическую голову.
        """
        # 1. Извлечение эмбеддинга
        emb = self.extractor.encode(text, normalize=True)  # [1, 384]

        # 2. Вероятности через классификационную голову
        probs = self.classifier.predict_proba(emb)[0]
        top_idx = int(np.argmax(probs))
        confidence = round(float(probs[top_idx]) * 100, 1)

        # 3. Косинусное сходство с прототипами (скалярное произведение нормализованных векторов)
        cosine_sims = np.dot(self.prototype_embeddings, emb.T).flatten()

        results_by_class = []
        for i in range(4):
            results_by_class.append({
                "index": i,
                "label": self.CATEGORIES[i],
                "emoji": self.CATEGORY_EMOJIS[i],
                "prob": round(float(probs[i]) * 100, 1),
                "similarity": round(float(cosine_sims[i]) * 100, 1)
            })

        results_by_class = sorted(results_by_class, key=lambda x: x["prob"], reverse=True)

        # 4. Анализ токенизации
        token_info = self.extractor.inspect_tokenization(text)

        # 5. Срез эмбеддинга (первые 10 чисел)
        vector_preview = " ".join([f"{v:+.3f}" for v in emb[0][:10]]) + " ... [384 dims]"

        # 6. Демонстрация Matryoshka Truncation (срез до 64 dims)
        matryoshka_64 = emb[0][:64]
        matryoshka_norm = matryoshka_64 / np.linalg.norm(matryoshka_64)

        return {
            "top_category": f"{self.CATEGORY_EMOJIS[top_idx]} {self.CATEGORIES[top_idx]}",
            "confidence": confidence,
            "classes": results_by_class,
            "token_info": token_info,
            "vector_preview": vector_preview,
            "matryoshka_preview": " ".join([f"{v:+.3f}" for v in matryoshka_norm[:6]]) + " ... [64 dims]"
        }
