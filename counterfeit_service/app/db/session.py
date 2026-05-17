"""
Async-сессия SQLAlchemy 2.x для PostgreSQL.

Используется и в main.py (FastAPI dependency get_db) и в worker.py
(прямое создание сессий через AsyncSessionLocal).
"""
import os

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://counterfeit:counterfeit@postgres:5432/counterfeit",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""


async def get_db():
    """FastAPI dependency: yields an async session per request."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Создаёт таблицы при старте сервиса (idempotent + concurrent-safe).

    Использует Postgres advisory lock, чтобы при одновременном старте API и
    worker'а только один процесс выполнял `CREATE TABLE ...`. Без блокировки
    второй процесс может попасть в окно между checkfirst-проверкой и
    insert'ом в pg_type, получив `duplicate key value violates unique
    constraint pg_type_typname_nsp_index` — функционально безвредно, но
    создаёт шум в логах и пугает оператора.
    """
    from sqlalchemy import text  # local import to keep top-level imports minimal
    # Импорт моделей нужен для того, чтобы Base.metadata их подхватил.
    from app.db import models  # noqa: F401
    is_postgres = engine.dialect.name == "postgresql"
    # Произвольная константа BIGINT для pg_advisory_lock. Один процесс берёт
    # блокировку → создаёт таблицы → отпускает; второй ждёт и видит уже
    # созданные таблицы (create_all с checkfirst=True их пропустит).
    LOCK_KEY = 7142852365876347597  # стабильный 64-битный идентификатор
    async with engine.begin() as conn:
        if is_postgres:
            await conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": LOCK_KEY})
        try:
            await conn.run_sync(Base.metadata.create_all)
        finally:
            if is_postgres:
                await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": LOCK_KEY})
