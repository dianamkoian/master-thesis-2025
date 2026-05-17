"""
Абстрактный контракт предиктора для сервиса детекции контрафакта.

Сервис спроектирован как pluggable: любая модель из исследовательской части
работы (M2 mainline, M2-FE+, ансамбли, R@P-optimal v2 и т. п.) может быть
подключена путём реализации `BasePredictor` без изменений `main.py` или
`worker.py`. См. документ `counterfeit_service/MODELS.md` для пошаговой
инструкции по подключению новой модели.

Все реализации обязаны соблюдать одинаковый контракт:

  - `load(self)`  — синхронная загрузка артефактов модели (CatBoost, скейлеры,
    энкодеры) с локального диска или удалённого хранилища.
  - `predict(self, image_bytes, name, description, brand, tab_inputs)`
    — синхронный инференс одного объекта по контракту, ожидаемому FastAPI
    эндпоинтом `/predict` и RabbitMQ-воркером.

Контракт возвращаемого словаря:
  {
    "is_counterfeit": bool,
    "probability":    float ∈ [0, 1],
    "signals": {
        "multimodal_score": float,
        "image_signal":     float,
        "text_signal":      float,
    },
  }

Поле `signals` сохраняется для совместимости с фронтендом и со схемой
`PredictionResponse` (`app/schemas.py`); если конкретная модель не различает
модальные сигналы, допустимо повторять там `multimodal_score`.
"""
from __future__ import annotations

import abc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BasePredictor(abc.ABC):
    """Базовый класс для всех имплементаций инференса.

    Подклассы:
      - `D2VCatBoostPredictor` (app/predictor.py) — текущая command-v1
        конфигурация (Doc2Vec + CLIP + CatBoost), сохранённая в
        `artifacts/catboost_model.cbm` + `d2v_model.pkl` + `img_scaler.pkl`.
        Используется по умолчанию.
      - `M2FeaturePlusPredictor` (заглушка, не реализована) — pipeline
        M2-FE+ из § 4.4.3.3 ВКР: 38 tab + CLIP + TF-IDF SVD + 36 FE-фич.
        Требует сохранения tfidf_vectorizer.pkl, svd.pkl, brand_stats.parquet,
        clip_centroids.npy, clip_kmeans.pkl + catboost_model_fe_plus.cbm.
      - `EnsemblePredictor` (заглушка) — взвешенное усреднение двух
        моделей по конфигу: $w \\cdot p_1 + (1-w) \\cdot p_2$.
    """

    #: Человекочитаемое имя реализации, видно в логах /health
    name: str = "base"

    #: Краткое описание модели (попадает в /health для отладки)
    description: str = ""

    @abc.abstractmethod
    def load(self) -> None:
        """Загрузить артефакты модели в память. Вызывается один раз при старте."""

    @abc.abstractmethod
    def predict(
        self,
        image_bytes: bytes,
        name: str,
        description: str,
        brand: str,
        tab_inputs: dict,
    ) -> dict:
        """Синхронный инференс одного объекта.

        Args:
            image_bytes: PNG/JPEG байты изображения товара.
            name:        `name_rus` карточки.
            description: `description` карточки.
            brand:       `brand_name` карточки.
            tab_inputs:  словарь {feature_name: value} с табличными признаками,
                         ожидаемыми моделью (включая `CommercialTypeName4`).

        Returns:
            Словарь по контракту, описанному в модуле docstring.
        """

    def health_info(self) -> dict:
        """Метаданные модели для эндпоинта `/health` (опционально)."""
        return {"predictor": self.name, "description": self.description}
