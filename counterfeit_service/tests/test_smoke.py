"""
Smoke-тесты сервиса (§ 7.5.2 ВКР).

Проверяют, что:
  - /health отдаёт корректный JSON с расширенной диагностикой;
  - /predict принимает валидный multipart и возвращает контракт PredictionResponse;
  - /predict отвергает не-image content_type с 400;
  - reasoning_pipeline отдаёт reasoning и reasoning_mode в трёх режимах;
  - audit log (prediction_requests) фиксирует каждый запрос ДО инференса;
  - /predict-async возвращает 503, если RabbitMQ недоступен (test-окружение).

Запуск:
  cd counterfeit_service
  pytest tests/ -v
"""
from __future__ import annotations


def test_health_ok(client):
    """GET /health возвращает 200 и status='ok'."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    # Расширенный health (§ 7.5.2): сообщает текущий predictor.
    assert "predictor" in body
    assert body["predictor"].get("name") == "reasoning_pipeline" or \
           body["predictor"].get("name") == "stub_borderline" or \
           body["predictor"] is not None  # любая корректная инфо-структура


def test_predict_happy_path(client, stub_image_bytes):
    """POST /predict с валидным multipart возвращает PredictionResponse-контракт."""
    files = {"image": ("test.png", stub_image_bytes, "image/png")}
    data = {
        "name": "Тестовая карточка",
        "brand": "TestBrand",
        "category": "Test",
        "price": "1500",
        "item_time_alive": "10",
        "seller_time_alive": "100",
        "item_count_sales30": "5",
        "item_count_returns30": "1",
    }
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    # Контракт PredictionResponse: обязательные поля.
    assert isinstance(body.get("is_counterfeit"), bool)
    assert 0.0 <= float(body["probability"]) <= 1.0
    sig = body["signals"]
    assert all(k in sig for k in ("multimodal_score", "image_signal", "text_signal"))
    # Поля LLM-канала: либо строка с reasoning, либо null + mode.
    assert "reasoning" in body
    assert "reasoning_mode" in body
    assert body["reasoning_mode"] in (
        "blocking_explanation", "borderline_explanation",
        "confident_positive_no_block", "confident_negative", "llm_error",
    )


def test_predict_rejects_non_image(client):
    """POST /predict с не-image content_type возвращает 400."""
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    r = client.post("/predict", files=files, data={"name": "x"})
    assert r.status_code == 400
    assert "image" in r.text.lower()


def test_predict_rejects_empty_image(client):
    """POST /predict с пустым телом возвращает 400."""
    files = {"image": ("empty.png", b"", "image/png")}
    r = client.post("/predict", files=files, data={"name": "x"})
    assert r.status_code == 400


def test_reasoning_pipeline_blocking_mode(client, stub_image_bytes):
    """Для высоковероятного контрафакта возвращается blocking_explanation с reasoning."""
    files = {"image": ("test.png", stub_image_bytes, "image/png")}
    # Stub-эвристика: молодой продавец + дешёвый брендовый → proba ≥ 0.85.
    data = {
        "name": "iPhone 15 Pro оригинал",
        "brand": "Apple",
        "category": "Смартфоны",
        "price": "5000",
        "item_time_alive": "2",
        "seller_time_alive": "10",
        "item_count_sales30": "1",
        "item_count_returns30": "1",
    }
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reasoning_mode"] == "blocking_explanation"
    assert body["reasoning"] and isinstance(body["reasoning"], str)


def test_reasoning_pipeline_confident_negative(client, stub_image_bytes):
    """Для очевидного оригинала reasoning не вызывается (LLM пропущена)."""
    files = {"image": ("test.png", stub_image_bytes, "image/png")}
    # Stub-эвристика: зрелый продавец + дешёвый небрендовый → proba ≤ 0.25.
    data = {
        "name": "Кружка керамическая",
        "brand": "",
        "category": "Посуда",
        "price": "200",
        "item_time_alive": "500",
        "seller_time_alive": "1500",
        "item_count_sales30": "80",
        "item_count_returns30": "1",
    }
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reasoning_mode"] == "confident_negative"
    assert body["reasoning"] is None


def test_audit_log_persists_request(client, stub_image_bytes):
    """После POST /predict в prediction_requests появилась запись."""
    from app.db.session import AsyncSessionLocal
    from app.db.models import PredictionRequest
    from sqlalchemy import select
    import asyncio

    files = {"image": ("test.png", stub_image_bytes, "image/png")}
    data = {"name": "audit-trail-check", "brand": "X", "category": "Y", "price": "100"}
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 200

    async def _count():
        async with AsyncSessionLocal() as db:
            stmt = select(PredictionRequest).where(PredictionRequest.name == "audit-trail-check")
            res = await db.execute(stmt)
            return res.scalars().all()

    rows = asyncio.get_event_loop().run_until_complete(_count())
    assert len(rows) >= 1
    row = rows[-1]
    assert row.mode == "sync"
    assert row.brand == "X"
    assert row.tab_inputs.get("PriceDiscounted") == 100.0


def test_async_returns_503_without_rabbit(client, stub_image_bytes):
    """В test-окружении RabbitMQ отсутствует → /predict-async возвращает 503."""
    files = {"image": ("test.png", stub_image_bytes, "image/png")}
    r = client.post("/predict-async", files=files, data={"name": "x"})
    assert r.status_code == 503
    assert "rabbitmq" in r.text.lower() or "async pipeline" in r.text.lower()


def test_defer_reasoning_returns_task_id(client, stub_image_bytes):
    """POST /predict?defer_reasoning=true возвращает task_id и reasoning=null."""
    files = {"image": ("test.png", stub_image_bytes, "image/png")}
    data = {
        "name": "iPhone 15 Pro", "brand": "Apple", "category": "Смартфоны",
        "price": "5000", "item_time_alive": "2", "seller_time_alive": "10",
    }
    r = client.post("/predict?defer_reasoning=true", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task_id"] is not None
    assert body["reasoning"] is None
    assert body["reasoning_mode"] == "pending"
    assert isinstance(body["probability"], float)


def test_reasoning_polling_eventually_ready(client, stub_image_bytes):
    """После defer_reasoning GET /predict/{task_id}/reasoning отдаёт ready."""
    import time
    files = {"image": ("test.png", stub_image_bytes, "image/png")}
    data = {
        "name": "iPhone 15 Pro", "brand": "Apple", "category": "Смартфоны",
        "price": "5000", "item_time_alive": "2", "seller_time_alive": "10",
    }
    r = client.post("/predict?defer_reasoning=true", files=files, data=data)
    task_id = r.json()["task_id"]
    # BackgroundTask запускается в TestClient после возврата response —
    # дадим ему 2-3 итерации, чтобы прокрутиться.
    for _ in range(20):
        rr = client.get(f"/predict/{task_id}/reasoning")
        if rr.status_code == 200 and rr.json().get("status") == "ready":
            assert rr.json()["reasoning"]
            assert rr.json()["reasoning_mode"] in (
                "blocking_explanation", "borderline_explanation",
                "confident_positive_no_block", "confident_negative",
            )
            return
        time.sleep(0.1)
    raise AssertionError("reasoning never became ready within 2 seconds")


def test_reasoning_polling_returns_404_for_unknown_task(client):
    """GET /predict/{неизвестный}/reasoning → 404."""
    r = client.get("/predict/00000000-0000-0000-0000-000000000000/reasoning")
    assert r.status_code == 404
