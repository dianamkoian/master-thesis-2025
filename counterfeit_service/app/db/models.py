"""
ORM-модели подсистемы хранения.

В рамках индивидуального вклада автора (§ 7.3) используются две таблицы:
  - `prediction_requests` — audit log входных запросов (raw input) для дебага и
    последующего retraining (§ 7.5);
  - `predictions_async` — асинхронные результаты инференса (RabbitMQ → worker).

Таблицы feedback и seller_profiles из подсистемы А. Бахтиаровой (§ 7.2)
здесь не описываются для минимизации зависимости подсистемы async-обработки
от прочих частей.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class PredictionRequest(Base):
    """
    Audit log входных API-запросов (sync /predict и async /predict-async).

    Хранит исходные параметры карточки в JSONB и относительный путь к
    сохранённому изображению. Используется для:
      - дебага «странных» предсказаний (input + result доступны для разбора);
      - сбора production-датасета для последующего retraining моделей;
      - аналитики (категории / диапазоны цен / возраст продавца / время суток).

    Связь с `predictions_async` — по полю `task_id` (для sync-запросов task_id
    генерируется автоматически и не публикуется в RabbitMQ; запись в этой
    таблице есть, в `predictions_async` — нет).
    """

    __tablename__ = "prediction_requests"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    mode: Mapped[str] = mapped_column(String(8), nullable=False)  # "sync" | "async"
    name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    brand: Mapped[str | None] = mapped_column(String(256), nullable=True)
    tab_inputs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    image_size_bytes: Mapped[int | None] = mapped_column(default=None, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class PredictionAsync(Base):
    """
    Результат асинхронного инференса, опубликованного через RabbitMQ.

    Worker записывает запись с status='done' либо 'error' после завершения
    задачи. API-эндпоинт GET /result/{task_id} читает эту таблицу.
    """

    __tablename__ = "predictions_async"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(16), default="processing", nullable=False)
    is_counterfeit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # § 4.4.9.7: LLM-канал. `reasoning` хранит текст объяснения от Qwen2.5
    # (для borderline и blocking режимов) или NULL (для confident_negative);
    # `reasoning_mode` — режим, по которому внешние CRM-системы могут
    # фильтровать записи (например, выгружать только blocking_explanation
    # для отправки селлерам / открытия тикетов в поддержке).
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning_mode: Mapped[str | None] = mapped_column(String(48), nullable=True)
    error: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
