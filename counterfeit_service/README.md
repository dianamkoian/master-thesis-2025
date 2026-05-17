# Counterfeit Detection Service

Прототип сервиса мультимодальной детекции контрафактных товаров на маркетплейсе Ozon. Часть магистерской ВКР Д. Мкоян (Глава 5 «Реализация»).

**Ключевые возможности:**
- **Pluggable predictor** (§ 5.2): любая исследовательская модель подключается через `PREDICTOR_TYPE` без правок API. Реализовано: `d2v_catboost` (deployed baseline), `reasoning_pipeline` (LLM-обёртка), `stub_borderline` (для тестов/демо). Заглушки: `m2_fe_plus`, `m2_fe_plus_mainline_ensemble` (см. `MODELS.md`).
- **Verdict-conditioned LLM-канал** (§ 3.4.9.7): Qwen2.5-1.5B-Instruct поверх любого primary predictor'а; четыре режима — `blocking_explanation` (≥ 0,85, для уведомления селлера), `confident_positive_no_block` (0,75–0,85, для QA), `borderline_explanation` (0,25–0,75, для модератора), `confident_negative` (≤ 0,25, без LLM).
- **Async-пайплайн** (§ 5.4, индивидуальный вклад автора): RabbitMQ durable + DLQ + PostgreSQL для batch-нагрузок.
- **Audit log** (§ 5.6): таблица `prediction_requests` фиксирует каждый входной запрос и ссылку на сохранённое изображение для дебага и retraining.
- **Расширенный `/health`**: статус всех компонентов (predictor + артефакты + БД + RabbitMQ + LLM) одним JSON.
- **Pytest smoke-набор**: 11 функциональных тестов (`test_smoke.py`) + 5 HCDM-сценариев (`test_hcdm_smoke.py`), без внешних зависимостей.

## Архитектура

```
                      ┌──────────────┐
                      │  Клиент / UI │
                      └──────┬───────┘
                             │
              POST /predict  │  POST /predict-async      GET /result/{task_id}
                             │
        ┌────────────────────┴─────────────────────────────────────────────┐
        │                    counterfeit-api (FastAPI)                     │
        │   sync ветка                  async ветка                        │
        │   ──────────                  ───────────                        │
        │   predictor.predict()         publish(task_id, payload)          │
        │        │                            │                            │
        │        ▼                            ▼                            │
        │   { result }                  RabbitMQ exchange                  │
        └────────────────────────────────────│─────────────────────────────┘
                                             │
                                             ▼
                                ┌────────────────────────┐
                                │   inference_queue      │
                                │   (durable, persistent)│
                                └────────────┬───────────┘
                                             │  consume (prefetch=1)
                            ┌────────────────┴────────────────┐
                            │   counterfeit-worker (× N)      │
                            │   aio_pika async consumer       │
                            │   CounterfeitPredictor          │
                            └────────────────┬────────────────┘
                                             │
                                             ▼
                                ┌────────────────────────┐
                                │  PostgreSQL (async)    │
                                │  predictions_async     │
                                └────────────────────────┘

                        ✗ ошибка инференса → inference_dlq (DLQ)
```

**Три модальности → Feature Fusion CatBoost** (общекомандный inference-модуль К. Азимовой):

```
Фото товара  →  CLIP ViT-B/32 (512-dim)  →  img_scaler  ─┐
Текст        →  Doc2Vec PV-DM (200-dim)                  ├─→  CatBoost 750-dim  →  {prob, signals}
Табличные    →  38 признаков продавца/товара             ─┘
```

## Структура

```
counterfeit_service/
├── app/
│   ├── main.py                # FastAPI: /predict (sync) и /predict-async (async)
│   ├── worker.py              # aio_pika consumer — § 5.4 Д. Мкоян
│   ├── predictor.py           # Feature Fusion inference (общекомандный, К. Азимова)
│   ├── schemas.py             # Pydantic models: sync + async ответы
│   └── db/                    # Async PostgreSQL (§ 5.4 + интеграция с § 5.3)
│       ├── session.py         # AsyncSessionLocal, engine, init_db
│       ├── models.py          # PredictionAsync
│       └── crud.py            # create_pending / mark_done / mark_error / get_by_task_id
├── static/
│   └── index.html             # Веб-интерфейс (С. Красовская, § 5.5)
├── artifacts/                 # Файлы моделей (см. «Артефакты»)
├── Dockerfile                 # Контейнер API
├── Dockerfile.worker          # Контейнер worker'а (тот же образ + другой CMD)
├── docker-compose.yml         # api + worker + rabbitmq + postgres
└── requirements.txt
```

## Артефакты моделей

Файлы моделей не хранятся в репозитории из-за большого размера.
Скачать с Яндекс Диска: https://disk.yandex.ru/d/aw6epg3MNkQ9vw

| Файл | Описание |
|---|---|
| `catboost_model.cbm` | Feature Fusion CatBoost (750 features) |
| `d2v_model.pkl` | Doc2Vec PV-DM 200-dim (≈ 555 МБ) |
| `img_scaler.pkl` | StandardScaler для CLIP-эмбеддингов |
| `feature_cols.pkl` | Список 38 табличных признаков |
| `cat_cols.pkl` | Категориальные признаки CatBoost |

После скачивания положить в `counterfeit_service/artifacts/`.

## Production-развёртывание

**Развёрнуто на Timeweb VPS:** **https://marketplace-fraud.ru** (TLS через Let's Encrypt с auto-renew, reverse-proxy Caddy v2).

```bash
# Health-check (production):
curl https://marketplace-fraud.ru/health

# Swagger UI:
open https://marketplace-fraud.ru/docs

# Web UI:
open https://marketplace-fraud.ru
```

Админ-интерфейсы (pgAdmin, RabbitMQ Management) на VPS слушают только `127.0.0.1` — публично не доступны. Подключение через SSH-туннель:

```bash
ssh -L 5050:localhost:5050 -L 15672:localhost:15672 root@5.129.242.72
# на Mac:
#   pgAdmin:           http://localhost:5050
#   RabbitMQ Mgmt UI:  http://localhost:15672 (login: counterfeit / <см. .env на VPS>)
```

Hot-swap при обновлении кода: `service/counterfeit_service/scripts/redeploy_timeweb.sh` (выполняется на VPS, без удаления volumes).

## Локальный запуск (для разработки)

### Полная сборка через Docker Compose (рекомендуется)

```bash
docker compose up --build
```

Поднимутся четыре контейнера:
- `counterfeit-rabbitmq` — брокер + management UI
- `counterfeit-postgres` — БД
- `counterfeit-api` — FastAPI
- `counterfeit-worker` — async consumer (можно масштабировать)

**Health-check (расширенный, § 5.6.2):**
```bash
curl http://localhost:8000/health | jq .
```
Возвращает:
```json
{
  "status": "ok",
  "version": "2.0.0",
  "predictor": {
    "name": "reasoning_pipeline",
    "description": "...",
    "inner_predictor": "d2v_catboost",
    "borderline_lo": 0.25, "borderline_hi": 0.75
  },
  "artifacts": {
    "catboost_model.cbm": {"size_bytes": 895360, "mtime": 1715470000},
    "d2v_model.pkl":      {"size_bytes": 458308819, ...}
  },
  "db": "ok",
  "rabbitmq": true,
  "llm": {
    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "device": "mps",
    "weights_loaded": false,
    "max_new_tokens": 120
  }
}
```

**Horizontal scaling worker'ов:**
```bash
docker compose up --build --scale counterfeit-worker=4
```

**RabbitMQ Management UI** (локальный запуск): http://localhost:15672 (login/password из локального `.env`). На production-VPS — через SSH-туннель (см. секцию выше).

### Локально (без Docker)

```bash
pip install -r requirements.txt
# Запустить RabbitMQ и PostgreSQL отдельно (или через docker compose up rabbitmq postgres)

# API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Worker — в отдельном терминале
python -m app.worker
```

Если RabbitMQ или PostgreSQL недоступны, API запустится без них; async-эндпоинты вернут `503`, а sync `/predict` будет работать.

## API

Открыть документацию: http://localhost:8000/docs (автоматически генерируемый Swagger UI).

### Sync: `POST /predict`

Возвращает результат сразу. Используется UI и быстрыми ad-hoc-проверками.

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@product.jpg" \
  -F "name=Картридж HP Q2612A" \
  -F "brand=NoName" \
  -F "category=Картриджи лазерные" \
  -F "price=250" \
  -F "item_time_alive=12" \
  -F "seller_time_alive=45"
```

Ответ:
```json
{
  "is_counterfeit": true,
  "probability": 0.87,
  "signals": {"multimodal_score": 0.87, "image_signal": 0.91, "text_signal": 0.43}
}
```

### Async: `POST /predict-async` + `GET /result/{task_id}`

Кладёт задачу в очередь, worker обрабатывает асинхронно.
Подходит для batch-нагрузок (десятки тысяч карточек в день).

```bash
# Шаг 1 — публикация в очередь
curl -X POST http://localhost:8000/predict-async \
  -F "image=@product.jpg" \
  -F "name=Картридж HP Q2612A" \
  ...
# → {"task_id":"3f8a...","status":"queued"}

# Шаг 2 — опрос результата
curl http://localhost:8000/result/3f8a...
# Сначала → {"status":"processing"}
# Через ~2 секунды → {"status":"done","is_counterfeit":true,"probability":0.87, ...}
```

## LLM-канал и persistence (§ 3.4.9.7)

Активация:
```bash
PREDICTOR_TYPE=reasoning_pipeline docker compose up -d
```

JSON-ответ расширяется обратно-совместимо двумя полями:
```json
{
  "is_counterfeit": true,
  "probability": 0.92,
  "signals": { "multimodal_score": 0.92, "image_signal": 0.5, "text_signal": 0.45 },
  "reasoning": "Карточка заблокирована: цена существенно ниже типичной для бренда, продавец зарегистрирован менее 30 дней назад.",
  "reasoning_mode": "blocking_explanation"
}
```

Поля сохраняются в `predictions_async`. CRM-системы фильтруют по `reasoning_mode`:
```sql
SELECT task_id, probability, reasoning, created_at
FROM predictions_async
WHERE reasoning_mode = 'blocking_explanation'
  AND created_at >= NOW() - INTERVAL '1 day';
```

Подробности — `MODELS.md`, раздел «Reasoning pipeline».

## Audit log (§ 5.6)

Каждый запрос к `/predict` и `/predict-async` фиксируется в таблице
`prediction_requests` ДО инференса; изображение сохраняется в директории
`AUDIT_IMAGE_STORAGE` (по умолчанию `/tmp/counterfeit_audit_images/YYYY-MM-DD/<task_id>.{png,jpg}`)
с соответствующим `image_path` в БД. Это позволяет:
- разобрать «странные» прогнозы (вход + результат доступны для аналитика);
- собрать production-датасет для retraining (раз в N дней — экспорт `prediction_requests ⨝ predictions_async`);
- проводить аналитику входов (распределение категорий, диапазоны цен, время суток).

Audit-сохранение реализовано как best-effort: сбой БД не ломает inference,
ошибка логируется, response клиенту не зависит от состояния `prediction_requests`.

## Тесты (§ 5.6.2)

```bash
PYTHONPATH=. pytest tests/ -v
```

11 функциональных тестов (`tests/test_smoke.py`), без внешних зависимостей:
in-memory SQLite, fake LLM через monkeypatch, отсутствующий RabbitMQ → 503.
Прогон ~1,1 секунды. Покрывают: `/health`, контракт `PredictionResponse`,
валидацию входа (400 на не-image и пустой файл), три режима reasoning,
audit log persistence, async fallback при отсутствии брокера.

Дополнительно `tests/test_hcdm_smoke.py` — 5 HCDM-сценариев (frozen-lookup
parquet, загрузка 58 410 probas, бит-точное сравнение, graceful fallback,
распределение proba); прогон ~19,6 секунды с реальными артефактами.

## Отказоустойчивость

- **Persistent очереди + сообщения** (`delivery_mode=PERSISTENT`, `durable=True`): задача не теряется при рестарте брокера.
- **Acknowledgement-семантика**: worker подтверждает обработку только после успешной записи в БД. При падении worker'а до ack задача автоматически переотправляется другому consumer'у.
- **Dead-letter queue** (`inference_dlq`): сообщения, на которых worker упал с исключением, отправляются туда без повторных попыток; пригодно для оператора-разборщика.
- **Авто-переподключение клиента**: `aio_pika.connect_robust` восстанавливает соединение при сбое сети без перезапуска процесса.
- **Graceful shutdown**: worker отрабатывает SIGTERM / SIGINT, заканчивает текущее сообщение и закрывает соединение.

## Развёртывание в облаке

Production-копия сервиса развёрнута в облачной среде Timeweb по адресу `http://5.129.242.72:8000` (Ubuntu 22.04 LTS, 4 ГБ RAM). См. § 5.6.4 ВКР для деталей деплоя.

## Распределение зон ответственности

| Компонент | Автор | Раздел ВКР |
|---|---|---|
| `app/main.py` (sync /predict, UI mount, CORS) | К. Азимова | § 5.2 |
| `app/predictor.py` (Feature Fusion inference) | К. Азимова | § 5.2 |
| `app/main.py` (async /predict-async + /result/{id}) | **Д. Мкоян** | **§ 5.4** |
| `app/worker.py` (aio_pika consumer) | **Д. Мкоян** | **§ 5.4** |
| `app/db/session.py`, `models.py`, `crud.py` | **Д. Мкоян** (на базе async SQLAlchemy от А. Бахтиаровой) | **§ 5.4** + § 5.3 |
| `Dockerfile.worker`, `docker-compose.yml` (rabbitmq + worker) | **Д. Мкоян** | **§ 5.4** |
| `static/index.html` (UI) | С. Красовская | § 5.5 |
| Деплой в Timeweb | С. Красовская | § 5.6.4 |
| Функциональное и нагрузочное тестирование | **Д. Мкоян** | **§ 5.6** |
