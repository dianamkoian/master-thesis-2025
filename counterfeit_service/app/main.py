"""
FastAPI-приложение сервиса детекции контрафакта.

Двухрежимная схема:

  POST /predict
       │
       └─► CounterfeitPredictor.predict() синхронно (как в командной v1.0).
           Используется UI и быстрыми ad-hoc проверками.

  POST /predict-async
       │
       ├─► publish задачи в RabbitMQ → возвращает { task_id, status: "queued" }
       │   Подходит для batch-нагрузок и масштабирования воркеров
       │   независимо от API.

  GET /result/{task_id}
       │
       └─► читает результат из PostgreSQL.

Подсистема async-обработки является индивидуальным вкладом Д. Мкоян (§ 7.3 ВКР);
синхронный эндпоинт /predict и инференс-модуль — общекомандный код
(К. Азимова, § 7.1).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aio_pika
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.session import get_db, init_db
from app.predictor import get_predictor
from app.reasoning_cache import reasoning_cache
from app.schemas import (
    AsyncResultResponse,
    PredictionResponse,
    ReasoningResponse,
    Signals,
    TaskQueuedResponse,
)
from app.storage import save_image

# Работает и локально (static/ рядом с app/), и в Docker (/app/static)
_HERE = Path(__file__).parent.parent  # counterfeit_service/
STATIC_DIR = str(os.getenv("STATIC_DIR", _HERE / "static"))
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://counterfeit:counterfeit@rabbitmq:5672/")
QUEUE_NAME = os.getenv("INFERENCE_QUEUE", "inference_queue")
DLQ_NAME = os.getenv("DEAD_LETTER_QUEUE", "inference_dlq")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

predictor = get_predictor()  # выбирается через env PREDICTOR_TYPE (default: d2v_catboost)

# Глобальные ссылки на RabbitMQ инициализируются в lifespan.
_rmq_connection: aio_pika.RobustConnection | None = None
_rmq_channel: aio_pika.RobustChannel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели + БД + RabbitMQ на старте, чистое завершение на выходе."""
    # Sync-инференс грузится всегда (нужен для /predict).
    predictor.load()

    # БД и брокер опциональны: если внешние сервисы недоступны,
    # async-эндпоинты вернут 503, sync-эндпоинт продолжит работать.
    global _rmq_connection, _rmq_channel
    try:
        await init_db()
        logger.info("PostgreSQL ready")
    except Exception as exc:
        logger.warning("init_db failed: %s — async pipeline disabled", exc)

    try:
        _rmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
        _rmq_channel = await _rmq_connection.channel()
        await _rmq_channel.declare_queue(DLQ_NAME, durable=True)
        await _rmq_channel.declare_queue(
            QUEUE_NAME,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": DLQ_NAME,
            },
        )
        logger.info("RabbitMQ ready, queues '%s' / '%s' declared", QUEUE_NAME, DLQ_NAME)
    except Exception as exc:
        logger.warning("RabbitMQ connect failed: %s — async pipeline disabled", exc)
        _rmq_connection = None
        _rmq_channel = None

    yield

    if _rmq_connection is not None:
        await _rmq_connection.close()


app = FastAPI(
    title="Counterfeit Detection Service",
    description="Multimodal counterfeit product detection for Ozon marketplace",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS allow_origins — берётся из env `CORS_ALLOWED_ORIGINS` (запятая как
# разделитель). Дефолт безопасный — только localhost; production-домены
# указываются в .env при деплое.
_cors_origins = [
    o.strip() for o in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,https://sonyakrasovskaya.github.io",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Базовые маршруты ──────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(Path(STATIC_DIR) / "index.html"))


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """Liveness + готовность всех компонентов (§ 7.5.2 ВКР).

    Возвращает структурированный отчёт по статусу основных подсистем:
      - status: "ok" если жив API (всегда 200, даже при деградации);
      - predictor: имя текущего predictor'а и его описание (для аудита);
      - artifacts: версии / размеры артефактов модели (для верификации деплоя);
      - db: ping PostgreSQL;
      - rabbitmq: доступность брокера;
      - llm: загружен ли Qwen2.5 (lazy: загружается при первом borderline-запросе).
    """
    from sqlalchemy import text as _sql_text
    health_info = getattr(predictor, "health_info", lambda: {"name": predictor.__class__.__name__})()

    # DB ping (через get_db — он уже сессия).
    db_status = "unknown"
    try:
        await db.execute(_sql_text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {type(exc).__name__}"

    # Артефакты модели (для предикторов, которые их используют).
    artifacts_info: dict[str, dict] = {}
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", _HERE / "artifacts"))
    if artifacts_dir.exists():
        for entry in artifacts_dir.iterdir():
            if entry.is_file() and not entry.name.startswith("."):
                artifacts_info[entry.name] = {
                    "size_bytes": entry.stat().st_size,
                    "mtime": int(entry.stat().st_mtime),
                }

    # LLM-канал: статус загрузки весов Qwen2.5 (lazy).
    llm_info: dict | None = None
    inner = getattr(predictor, "llm", None)
    if inner is not None:
        llm_info = {
            "model_path": getattr(inner, "model_path", None),
            "device": getattr(inner, "device", None),
            "weights_loaded": getattr(inner, "model", None) is not None,
            "max_new_tokens": getattr(inner, "max_new_tokens", None),
        }

    return {
        "status": "ok",
        "version": app.version,
        "predictor": health_info,
        "artifacts": artifacts_info,
        "db": db_status,
        "rabbitmq": _rmq_channel is not None,
        "llm": llm_info,
    }


# ─── Synchronous /predict (общекомандный, К. Азимова) ─────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Product photo"),
    name: str = Form("", description="Product name (name_rus)"),
    description: str = Form("", description="Product description"),
    brand: str = Form("", description="Brand name"),
    category: str = Form("", description="CommercialTypeName4 — product category string"),
    price: Optional[float] = Form(None, description="PriceDiscounted; omit if unknown"),
    item_time_alive: Optional[float] = Form(None, description="Days on marketplace; omit if unknown"),
    item_count_sales30: Optional[float] = Form(None, description="Sales last 30 days; omit if unknown"),
    item_count_returns30: Optional[float] = Form(None, description="Returns last 30 days; omit if unknown"),
    seller_time_alive: Optional[float] = Form(None, description="Seller age in days; omit if unknown"),
    item_id: Optional[int] = Form(
        None,
        description="Опциональный идентификатор товара (`id` или `ItemID`). "
                    "При совпадении с канонической позицией тестового сплита (n = 58 410) "
                    "predictor `cdsm_v3_4channel` возвращает headline-вероятность "
                    "из § 5.4.6.6 ВКР; иначе — graceful fallback на baseline.",
    ),
    defer_reasoning: bool = Query(
        False,
        description="Progressive disclosure (§ 7.4): если true, вернуть быстрый "
                    "verdict без LLM-объяснения; клиент опрашивает "
                    "GET /predict/{task_id}/reasoning отдельно.",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Синхронный инференс. Возвращает результат сразу (≈ 1–2 секунды)."""
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    tab_inputs = _build_tab_inputs(
        category, price, item_time_alive, item_count_sales30,
        item_count_returns30, seller_time_alive, item_id=item_id,
    )

    # Audit log: input → БД ДО инференса, чтобы остался след даже при сбое модели.
    task_id = str(uuid.uuid4())
    image_path = save_image(task_id, image_bytes, image.content_type)
    try:
        await crud.log_request(
            db,
            task_id=task_id, mode="sync",
            name=name, description=description, brand=brand,
            tab_inputs=tab_inputs,
            image_path=image_path, image_size_bytes=len(image_bytes),
        )
    except Exception:  # noqa: BLE001
        # БД-сбой не должен ронять inference; audit log — best-effort.
        logger.exception("Audit log failed for sync request task_id=%s", task_id)

    try:
        if defer_reasoning and hasattr(predictor, "inner"):
            # Двухстадийный отклик: вычислить только вердикт (без LLM),
            # запустить генерацию reasoning в BackgroundTask.
            inner = predictor.inner
            result = inner.predict(image_bytes, name, description, brand, tab_inputs)
            result.setdefault("reasoning", None)
            result["reasoning_mode"] = "pending"
            result["task_id"] = task_id
            reasoning_cache.mark_pending(task_id)
            background_tasks.add_task(
                _generate_reasoning_background,
                task_id, predictor, result, name, description, brand, tab_inputs,
            )
        else:
            result = predictor.predict(
                image_bytes=image_bytes, name=name, description=description,
                brand=brand, tab_inputs=tab_inputs,
            )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # Сохраняем sync-результат в БД так же, как async, чтобы он попал в
    # /api/metrics (UI-страница «Метрики» и downstream CRM SELECT-ы по
    # signals->>'model_route' одинаково работают для обоих режимов).
    # Если defer_reasoning=true, reasoning ещё генерируется в фоне —
    # сохраняем без него, BackgroundTask дополнит запись.
    try:
        await crud.mark_done(
            db,
            task_id=task_id,
            is_counterfeit=bool(result.get("is_counterfeit", False)),
            probability=float(result.get("probability", 0.0)),
            signals=result.get("signals", {}),
            reasoning=result.get("reasoning"),
            reasoning_mode=result.get("reasoning_mode"),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist sync result task_id=%s", task_id)

    return PredictionResponse(**result)


def _generate_reasoning_background(
    task_id: str,
    predictor_obj,
    inner_result: dict,
    name: str,
    description: str,
    brand: str,
    tab_inputs: dict,
) -> None:
    """Background task: формирует reasoning через template-канал по умолчанию
    или LLM-канал, если он включён в predictor_obj.
    """
    try:
        proba = float(inner_result.get("probability", 0.0))
        lo = getattr(predictor_obj, "borderline_lo", 0.25)
        hi = getattr(predictor_obj, "borderline_hi", 0.75)
        block = getattr(predictor_obj, "blocking_threshold", 0.85)
        if proba >= block:
            mode = "blocking_explanation"
        elif lo < proba < block:
            mode = "borderline_explanation" if proba < hi else "confident_positive_no_block"
        else:
            mode = "confident_negative"
        if mode == "confident_negative":
            reasoning_cache.mark_ready(task_id, "", mode)
            return
        card = {"name": name, "description": description, "brand": brand, **(tab_inputs or {})}
        if getattr(predictor_obj, "llm", None) is not None:
            text = predictor_obj.llm.explain(card, proba, inner_result.get("signals"), mode=mode)
        else:
            from app.template_reasoner import generate_reasoning
            text = generate_reasoning(card, proba, mode=mode)
        reasoning_cache.mark_ready(task_id, text, mode)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Background reasoning failed for task_id=%s", task_id)
        reasoning_cache.mark_error(task_id, str(exc))


@app.get("/predict/{task_id}/reasoning", response_model=ReasoningResponse)
async def get_reasoning(task_id: str):
    """Polling-эндпоинт для progressive disclosure (§ 7.4).

    Возвращает текущее состояние LLM-генерации:
      - 200 + status='ready' с reasoning'ом, когда LLM закончила;
      - 202 + status='pending', пока идёт генерация — клиент опрашивает повторно;
      - 200 + status='error', если LLM упала;
      - 404, если task_id неизвестен или истёк TTL.
    """
    entry = reasoning_cache.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"task_id {task_id} not found or expired")
    resp = ReasoningResponse(
        task_id=task_id,
        status=entry.status,
        reasoning=entry.reasoning,
        reasoning_mode=entry.reasoning_mode,
        error=entry.error,
    )
    # FastAPI возвращает 200 по умолчанию; для pending переключаем на 202.
    if entry.status == "pending":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=202, content=resp.model_dump())
    return resp


# ─── Asynchronous /predict-async + /result/{id} (Д. Мкоян, § 7.3) ─

@app.post("/predict-async", response_model=TaskQueuedResponse)
async def predict_async(
    image: UploadFile = File(..., description="Product photo"),
    name: str = Form(""),
    description: str = Form(""),
    brand: str = Form(""),
    category: str = Form(""),
    price: Optional[float] = Form(None),
    item_time_alive: Optional[float] = Form(None),
    item_count_sales30: Optional[float] = Form(None),
    item_count_returns30: Optional[float] = Form(None),
    seller_time_alive: Optional[float] = Form(None),
    item_id: Optional[int] = Form(
        None,
        description="Опциональный идентификатор товара для headline-режима "
                    "`cdsm_v3_4channel` (см. /predict).",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Публикует задачу инференса в очередь RabbitMQ и возвращает task_id.
    Worker-процесс асинхронно обрабатывает задачу и сохраняет результат
    в PostgreSQL. Клиент опрашивает результат через GET /result/{task_id}.
    """
    if _rmq_channel is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Async pipeline unavailable: RabbitMQ not connected. "
                "Use /predict for synchronous inference."
            ),
        )

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    task_id = str(uuid.uuid4())
    tab_inputs = _build_tab_inputs(
        category, price, item_time_alive, item_count_sales30,
        item_count_returns30, seller_time_alive, item_id=item_id,
    )

    # 1) Audit log: сохраняем исходный запрос и изображение до publish.
    image_path = save_image(task_id, image_bytes, image.content_type)
    try:
        await crud.log_request(
            db,
            task_id=task_id, mode="async",
            name=name, description=description, brand=brand,
            tab_inputs=tab_inputs,
            image_path=image_path, image_size_bytes=len(image_bytes),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Audit log failed for async request task_id=%s", task_id)

    # 2) Маркируем processing-запись в БД до publish, чтобы избежать race.
    try:
        await crud.create_pending(db, task_id)
    except Exception as exc:
        logger.warning("DB create_pending failed: %s — продолжаем без записи", exc)

    # 3) Публикуем durable persistent-сообщение.
    payload = {
        "task_id": task_id,
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "name": name,
        "description": description,
        "brand": brand,
        "tab_inputs": tab_inputs,
    }

    await _rmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
            message_id=task_id,
        ),
        routing_key=QUEUE_NAME,
    )

    logger.info("Queued task_id=%s", task_id)
    return TaskQueuedResponse(task_id=task_id, status="queued")


@app.get("/result/{task_id}", response_model=AsyncResultResponse)
async def get_result(task_id: str, db: AsyncSession = Depends(get_db)):
    """Возвращает текущий статус задачи + результат, когда он готов."""
    record = await crud.get_by_task_id(db, task_id)

    if record is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    signals = Signals(**record.signals) if record.signals else None
    return AsyncResultResponse(
        task_id=record.task_id,
        status=record.status,
        is_counterfeit=record.is_counterfeit,
        probability=record.probability,
        signals=signals,
        reasoning=record.reasoning,
        reasoning_mode=record.reasoning_mode,
        error=record.error,
    )


# ─── Metrics dashboard (для UI-страницы «Метрики», § 7.5 ВКР) ─────

@app.get("/api/metrics", include_in_schema=False)
async def api_metrics(db: AsyncSession = Depends(get_db)):
    """Агрегаты по БД для UI-страницы метрик.

    Возвращает: общие счётчики, разбивку по mode/route/verdict/reasoning_mode,
    распределение probability, последние 20 предсказаний. UI обновляет страницу
    по этому эндпоинту без перезагрузки. Не для рецензента-API (включён в /docs
    отключён через `include_in_schema=False`), это internal dashboard.
    """
    from sqlalchemy import text
    out: dict = {}

    # 1. Общие счётчики prediction_requests
    res = await db.execute(text("""
        SELECT mode, COUNT(*) AS n
        FROM prediction_requests
        GROUP BY mode
    """))
    out['requests_by_mode'] = {r.mode: r.n for r in res.fetchall()}
    out['total_requests'] = sum(out['requests_by_mode'].values())

    # 2. Async по статусу
    res = await db.execute(text("""
        SELECT status, COUNT(*) AS n
        FROM predictions_async GROUP BY status
    """))
    out['async_by_status'] = {r.status: r.n for r in res.fetchall()}

    # 3. Async по model_route
    res = await db.execute(text("""
        SELECT
            COALESCE(signals->>'model_route', 'unknown') AS route,
            COUNT(*) AS n,
            ROUND(AVG(probability)::numeric, 4) AS avg_proba,
            SUM(CASE WHEN is_counterfeit THEN 1 ELSE 0 END) AS fake_count
        FROM predictions_async
        WHERE status='done'
        GROUP BY route
        ORDER BY n DESC
    """))
    out['by_route'] = [
        {'route': r.route, 'count': r.n, 'avg_proba': float(r.avg_proba),
         'fake_count': r.fake_count}
        for r in res.fetchall()
    ]

    # 4. Распределение reasoning_mode
    res = await db.execute(text("""
        SELECT COALESCE(reasoning_mode, 'null') AS mode, COUNT(*) AS n
        FROM predictions_async WHERE status='done'
        GROUP BY mode ORDER BY n DESC
    """))
    out['reasoning_modes'] = [{'mode': r.mode, 'count': r.n} for r in res.fetchall()]

    # 5. Гистограмма probability (10 bins по 0,1)
    res = await db.execute(text("""
        SELECT
            FLOOR(probability * 10)::int AS bin,
            COUNT(*) AS n
        FROM predictions_async WHERE status='done' AND probability IS NOT NULL
        GROUP BY bin ORDER BY bin
    """))
    out['proba_histogram'] = [{'bin': r.bin, 'count': r.n} for r in res.fetchall()]

    # 6. Последние 20 предсказаний
    res = await db.execute(text("""
        SELECT
            SUBSTRING(p.task_id, 1, 8) AS task,
            COALESCE(r.mode, 'async') AS mode,
            r.name,
            r.brand,
            r.tab_inputs->>'CommercialTypeName4' AS category,
            r.tab_inputs->>'PriceDiscounted' AS price,
            r.tab_inputs->>'ItemID' AS item_id,
            p.is_counterfeit,
            p.probability,
            p.signals->>'model_route' AS route,
            p.reasoning_mode,
            p.updated_at
        FROM predictions_async p
        LEFT JOIN prediction_requests r ON r.task_id = p.task_id
        WHERE p.status='done'
        ORDER BY p.updated_at DESC LIMIT 20
    """))
    out['recent'] = [
        {
            'task': r.task,
            'mode': r.mode,
            'name': r.name or '',
            'brand': r.brand or '',
            'category': r.category or '',
            'price': r.price,
            'item_id': r.item_id,
            'is_counterfeit': r.is_counterfeit,
            'probability': float(r.probability) if r.probability is not None else None,
            'route': r.route or '',
            'reasoning_mode': r.reasoning_mode or '',
            'time': r.updated_at.strftime('%H:%M:%S') if r.updated_at else '',
        }
        for r in res.fetchall()
    ]

    # 7. Метрики из ВКР для сравнения (статичные)
    out['thesis_metrics'] = {
        'pr_auc': 0.8044, 'roc_auc': 0.9720, 'r_at_p09': 0.2068,
        'pr_auc_clean_ei': 0.7909, 'r_at_p09_clean_ei': 0.3920,
        'n_test': 58410, 'n_clean_ei': 24842,
        'multi_seed_significant': '5/5',
    }
    return out


# ─── Helpers ───────────────────────────────────────────────────────

def _build_tab_inputs(
    category: str,
    price: Optional[float],
    item_time_alive: Optional[float],
    item_count_sales30: Optional[float],
    item_count_returns30: Optional[float],
    seller_time_alive: Optional[float],
    item_id: Optional[int] = None,
) -> dict:
    """Собирает словарь tabular-признаков для predictor.predict().

    `None` для числовых полей означает «пользователь оставил поле пустым»
    (= признак отсутствует). Этот сигнал missingness прокидывается в
    predictor as-is — внутри канала каждый predictor сам решает, чем
    заполнять пропуск (Mode 3 fillna(-1), M2-FE+ fillna(0), etc.) согласно
    своему training-контракту. Раньше здесь была склейка None → 0.0,
    которая нарушала Mode 3 fillna(-1) и сглаживала сигнал missingness.

    Если пользователь хочет передать ноль явно — он вводит `0` в поле UI,
    и это попадает как `0.0` (отличается от None).

    `item_id` пробрасывается под двумя ключами (`id` и `ItemID`) — это позволяет
    headline-предиктору найти каноническую вероятность в lookup-таблице
    тестового сплита без амбигуитета.
    """
    payload: dict = {
        "CommercialTypeName4": category,
        "PriceDiscounted": price,
        "item_time_alive": item_time_alive,
        "item_count_sales30": item_count_sales30,
        "item_count_returns30": item_count_returns30,
        "seller_time_alive": seller_time_alive,
    }
    if item_id is not None:
        payload["id"] = int(item_id)
        payload["ItemID"] = int(item_id)
    return payload
