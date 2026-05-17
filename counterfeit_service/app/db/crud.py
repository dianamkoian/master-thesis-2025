"""
CRUD-операции над PredictionAsync и PredictionRequest.

Все функции async, используют AsyncSession (SQLAlchemy 2.x).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PredictionAsync, PredictionRequest


async def log_request(
    db: AsyncSession,
    *,
    task_id: str,
    mode: str,
    name: str,
    description: str,
    brand: str,
    tab_inputs: dict,
    image_path: str | None,
    image_size_bytes: int,
) -> None:
    """Сохранить audit-log входного запроса.

    Вызывается из /predict (mode='sync') и /predict-async (mode='async') до
    публикации задачи; гарантирует, что любая accepted-запросная единица
    оставляет след вне зависимости от исхода инференса. См. § 7.5 ВКР.
    """
    record = PredictionRequest(
        task_id=task_id,
        mode=mode,
        name=name[:512] if name else None,
        description=description if description else None,
        brand=brand[:256] if brand else None,
        tab_inputs=tab_inputs or {},
        image_path=image_path,
        image_size_bytes=image_size_bytes,
    )
    db.add(record)
    await db.commit()


async def get_request(db: AsyncSession, task_id: str) -> PredictionRequest | None:
    stmt = select(PredictionRequest).where(PredictionRequest.task_id == task_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def create_pending(db: AsyncSession, task_id: str) -> PredictionAsync:
    """Создаёт запись со status='processing' в момент публикации задачи в очередь."""
    record = PredictionAsync(task_id=task_id, status="processing")
    db.add(record)
    await db.commit()
    return record


async def get_by_task_id(db: AsyncSession, task_id: str) -> PredictionAsync | None:
    """Возвращает запись по task_id либо None."""
    stmt = select(PredictionAsync).where(PredictionAsync.task_id == task_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def mark_done(
    db: AsyncSession,
    task_id: str,
    is_counterfeit: bool,
    probability: float,
    signals: dict,
    reasoning: str | None = None,
    reasoning_mode: str | None = None,
) -> None:
    """Помечает задачу как завершённую и сохраняет результат инференса.

    Параметры `reasoning` и `reasoning_mode` заполняются predictor'ом
    `reasoning_pipeline` (§ 4.4.9.7); для других predictor'ов остаются None.
    Внешние CRM-системы фильтруют по `reasoning_mode` (например,
    `WHERE reasoning_mode = 'blocking_explanation'` — выгрузка под автоблок).
    """
    record = await get_by_task_id(db, task_id)
    if record is None:
        record = PredictionAsync(task_id=task_id)
        db.add(record)
    record.status = "done"
    record.is_counterfeit = is_counterfeit
    record.probability = probability
    record.signals = signals
    record.reasoning = reasoning
    record.reasoning_mode = reasoning_mode
    record.error = None
    await db.commit()


async def mark_error(db: AsyncSession, task_id: str, error_msg: str) -> None:
    """Помечает задачу как завершённую с ошибкой."""
    record = await get_by_task_id(db, task_id)
    if record is None:
        record = PredictionAsync(task_id=task_id)
        db.add(record)
    record.status = "error"
    record.error = error_msg[:512]
    await db.commit()
