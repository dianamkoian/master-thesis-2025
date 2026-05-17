"""
RabbitMQ-consumer для асинхронной обработки задач инференса.

Логика:
  1. Подключаемся к RabbitMQ (aio_pika.connect_robust — авто-переподключение).
  2. Декларируем те же queues, что и API (durable + dead-letter routing).
  3. Каждое сообщение обрабатывается атомарно: decode → predict → save → ack.
  4. Любая ошибка во время инференса → status='error' в БД + reject без
     повторной доставки (сообщение уходит в DLQ для разбора оператором).
  5. prefetch_count = 1 — каждый worker берёт по одной задаче за раз,
     что предсказуемо при CPU-bound инференсе.

Запуск:
    python -m app.worker

Индивидуальный вклад Д. Мкоян (§ 7.3 ВКР).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
from typing import Optional

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from app.db import crud
from app.db.session import AsyncSessionLocal, init_db
from app.predictor import get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] worker: %(message)s",
)
logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://counterfeit:counterfeit@rabbitmq:5672/")
QUEUE_NAME = os.getenv("INFERENCE_QUEUE", "inference_queue")
DLQ_NAME = os.getenv("DEAD_LETTER_QUEUE", "inference_dlq")
PREFETCH = int(os.getenv("WORKER_PREFETCH", "1"))

# Predictor загружается один раз при старте процесса.
predictor = get_predictor()  # выбирается через env PREDICTOR_TYPE (default: d2v_catboost)


async def _handle_message(message: AbstractIncomingMessage) -> None:
    """Обрабатывает одно сообщение из очереди inference_queue."""
    task_id: Optional[str] = None
    try:
        payload = json.loads(message.body.decode("utf-8"))
        task_id = payload["task_id"]
        logger.info("Processing task_id=%s", task_id)

        image_bytes = base64.b64decode(payload["image"])
        result = predictor.predict(
            image_bytes=image_bytes,
            name=payload.get("name", ""),
            description=payload.get("description", ""),
            brand=payload.get("brand", ""),
            tab_inputs=payload.get("tab_inputs", {}),
        )

        async with AsyncSessionLocal() as db:
            await crud.mark_done(
                db,
                task_id=task_id,
                is_counterfeit=bool(result["is_counterfeit"]),
                probability=float(result["probability"]),
                signals=result["signals"],
                reasoning=result.get("reasoning"),
                reasoning_mode=result.get("reasoning_mode"),
            )

        await message.ack()
        logger.info("Done task_id=%s probability=%.4f", task_id, result["probability"])

    except Exception as exc:
        logger.exception("Failed task_id=%s: %s", task_id, exc)
        if task_id is not None:
            try:
                async with AsyncSessionLocal() as db:
                    await crud.mark_error(db, task_id, str(exc))
            except Exception:
                logger.exception("Failed to record error in DB for task_id=%s", task_id)
        # Не реквеуем — отправляем в DLQ через x-dead-letter (см. main.py)
        await message.reject(requeue=False)


async def _run() -> None:
    """Запускает consumer-loop. Восстанавливается при сбое соединения."""
    predictor.load()
    logger.info("Predictor loaded")

    # Гарантируем, что таблицы существуют (worker может стартовать раньше API).
    try:
        await init_db()
    except Exception as exc:
        logger.warning("init_db failed: %s — продолжаем (схема должна быть создана API)", exc)

    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL, timeout=5)
    except Exception as exc:  # noqa: BLE001
        bar = "─" * 72
        msg = (
            f"\n{bar}\n"
            f"  ✗ Не удалось подключиться к RabbitMQ ({type(exc).__name__}).\n"
            f"    URL:    {RABBITMQ_URL}\n"
            f"    Ошибка: {exc}\n"
            f"\n"
            f"  Возможные причины и решения:\n"
            f"    1. RabbitMQ не запущен. Поднять брокер + БД:\n"
            f"         docker compose up -d rabbitmq postgres\n"
            f"       (затем проверить: docker ps | grep rabbitmq)\n"
            f"    2. Hostname 'rabbitmq' — это имя docker-compose сервиса,\n"
            f"       оно не резолвится вне docker-сети. Для локального запуска:\n"
            f"         export RABBITMQ_URL='amqp://counterfeit:counterfeit@localhost:5672/'\n"
            f"    3. Если нужен только sync /predict без async — worker\n"
            f"       не нужен вовсе, достаточно `uvicorn app.main:app`.\n"
            f"{bar}"
        )
        logger.error(msg)
        raise SystemExit(1)

    channel = await connection.channel()
    await channel.set_qos(prefetch_count=PREFETCH)

    # Дублируем декларацию queues — на случай если worker стартовал первым.
    await channel.declare_queue(DLQ_NAME, durable=True)
    queue = await channel.declare_queue(
        QUEUE_NAME,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": DLQ_NAME,
        },
    )

    logger.info("Consumer ready, listening on '%s' (prefetch=%d)", QUEUE_NAME, PREFETCH)

    stop_event = asyncio.Event()

    def _request_shutdown(*_: object) -> None:
        logger.info("Shutdown signal received, draining...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _request_shutdown)

    async with queue.iterator() as consumer:
        async for message in consumer:
            await _handle_message(message)
            if stop_event.is_set():
                break

    await connection.close()
    logger.info("Worker stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(_run())
