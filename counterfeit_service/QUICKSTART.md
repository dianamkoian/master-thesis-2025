# Quick-start: запустить сервис локально

> **Production-инстанс:** **https://marketplace-fraud.ru** (HTTPS, Let's Encrypt). Локальный запуск нужен только для разработки и отладки — для демонстрации и проверки достаточно prod-инстанса.

## Однократная подготовка

Нужно установлено: **Docker Desktop** (на macOS — из Applications), активен (зелёная иконка в трее).

## Запуск сервиса

```bash
cd /Users/diana/master-thesis-2026/service/counterfeit_service
docker compose up -d
```

Первый запуск с нуля: ~6 минут (билд образов + загрузка CLIP). Повторный запуск: ~10 секунд.

## Проверка работоспособности

```bash
# 1. Все контейнеры подняты
docker compose ps
# → counterfeit-api, counterfeit_service-counterfeit-worker-1, counterfeit-postgres, counterfeit-rabbitmq

# 2. API готов (опрос до 200)
until curl -sf http://localhost:8000/health >/dev/null; do sleep 2; done && echo "API ready"

# 3. /health возвращает inner_predictor=hcdm_4channel
curl -s http://localhost:8000/health | python3 -m json.tool | head -10
```

## Откройте UI

В браузере: **http://localhost:8000**

Демо-карточка:
- Название: `Samsung Galaxy S24`
- Бренд: `Samsung`
- Категория: `Смартфоны`
- Цена: `65000`
- Дней на площадке: `120`, Продажи 30д: `45`, Возвраты: `2`, Возраст продавца: `850`
- ItemID: оставить пустым → live HCDM, или `202599` → frozen lookup из § 5.4.6

## Проверка HCDM путей (для защиты)

```bash
# Frozen lookup: canonical ItemID 202599 → бит-точное HCDM-значение
curl -s -X POST http://localhost:8000/predict \
  -F "image=@/tmp/test_img.png" \
  -F "name=test" -F "brand=Samsung" -F "category=Смартфоны" -F "price=65000" \
  -F "item_id=202599" | python3 -m json.tool | grep -E "probability|model_route"
# → "probability": 0.006111432798206806, "model_route": "hcdm_4channel_headline"

# Live HCDM: unknown ItemID → 4-канальная композиция
curl -s -X POST http://localhost:8000/predict \
  -F "image=@/tmp/test_img.png" \
  -F "name=iPhone 16" -F "brand=Apple" -F "category=Смартфоны" -F "price=89990" \
  -F "item_id=999999" | python3 -m json.tool | grep -E "probability|p_social|p_mobile|p_real|p_fin|model_route"
# → "model_route": "hcdm_4channel_live" + 4 канала
```

Тестовое изображение (любой PNG 224×224):
```bash
python3 -c "from PIL import Image; Image.new('RGB',(224,224),(255,255,255)).save('/tmp/test_img.png')"
```

## Логи и мониторинг

```bash
docker logs counterfeit-api --tail 30        # API
docker logs counterfeit_service-counterfeit-worker-1 --tail 30   # Worker
docker logs counterfeit-rabbitmq --tail 30   # RabbitMQ
docker logs counterfeit-postgres --tail 30   # Postgres
```

RabbitMQ UI (локально): **http://localhost:15672** (login/password из локального `.env`).

На production-VPS pgAdmin и RabbitMQ Mgmt UI слушают только `127.0.0.1` — подключение через SSH-туннель:
```bash
ssh -L 5050:localhost:5050 -L 15672:localhost:15672 root@5.129.242.72
# затем у себя: http://localhost:5050 (pgAdmin), http://localhost:15672 (RabbitMQ)
```

## Остановить / перезапустить

```bash
docker compose stop                          # стоп без удаления (быстрый старт)
docker compose down                          # стоп с удалением контейнеров (volumes остаются)
docker compose up -d --force-recreate counterfeit-api counterfeit-worker  # пересоздать только API+worker
docker compose build counterfeit-api counterfeit-worker                   # пересобрать образ (после изменения кода)
```

## Если что-то не так

- **API не стартует**: `docker logs counterfeit-api --tail 50` — обычно либо отсутствует artifact в `artifacts/`, либо упал DB/RabbitMQ. Проверь `docker compose ps`.
- **Изменила код app/**: нужна пересборка → `docker compose build counterfeit-api counterfeit-worker && docker compose up -d --force-recreate counterfeit-api counterfeit-worker`.
- **Изменила только static/**: монтируется напрямую, достаточно обновить страницу в браузере (если static подключён через volume — посмотри `docker-compose.yml`; в текущей конфигурации static встроен в образ, поэтому нужна пересборка).
- **`model_route=hcdm_headline_fallback_baseline`** вместо HCDM live: пропал какой-то артефакт в `artifacts/cdsm_v3/`. Проверь `ls artifacts/cdsm_v3/*.pkl artifacts/cdsm_v3/*.cbm artifacts/cdsm_v3/*.joblib`.

## Деплой на VPS

См. `DEPLOY_TIMEWEB.md` — отдельная инструкция по развёртыванию на VPS.
