# Counterfeit Detection Service

Прототип сервиса мультимодальной детекции контрафактных товаров на маркетплейсе Ozon.

## Архитектура

Три модальности → Feature Fusion CatBoost:

```
Фото товара  →  CLIP ViT-B/32 (512-dim)  →  img_scaler
Текст        →  Doc2Vec PV-DM (200-dim)
Табличные    →  38 признаков продавца/товара
                        ↓
              CatBoost Fusion Model
                        ↓
         { is_counterfeit, probability, signals }
                        ↓
                  PostgreSQL DB
```

**Метрики модели:** ROC-AUC 0.9228 | PR-AUC 0.6665 | Recall@P≥0.9 0.0206

## Структура

```
counterfeit_service/
├── app/
│   ├── main.py        # FastAPI роуты
│   ├── predictor.py   # Inference pipeline
│   ├── schemas.py     # Pydantic схемы
│   ├── database.py    # SQLAlchemy engine, сессия
│   └── models.py      # ORM-модели (predictions, feedback, seller_profiles)
├── static/
│   └── index.html     # Веб-интерфейс
├── artifacts/         # Файлы моделей (скачать отдельно, см. ниже)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## База данных

Сервис использует PostgreSQL для хранения истории предсказаний и агрегатов по продавцам.

| Таблица | Назначение | Ключевые поля |
|---------|-----------|---------------|
| `predictions` | каждый вызов `/predict` | probability, signals, входные данные |
| `feedback` | оценка модератора | correct, true_label, prediction_id |
| `seller_profiles` | агрегат по продавцу | flagged_count, avg_probability |

База поднимается автоматически через Docker Compose. Таблицы создаются при первом запуске.

## Артефакты моделей

Файлы моделей не хранятся в репозитории из-за большого размера.  
Скачать с Яндекс Диска: https://disk.yandex.ru/d/aw6epg3MNkQ9vw

| Файл | Описание |
|------|----------|
| `catboost_model.cbm` | Feature Fusion CatBoost (750 features) |
| `img_scaler.pkl` | StandardScaler для CLIP эмбеддингов |
| `feature_cols.pkl` | Список 38 табличных признаков |
| `cat_cols.pkl` | Категориальные признаки CatBoost |
| `d2v_model.pkl` | Doc2Vec PV-DM 200-dim (~555 MB) — необязательно |

> Без `d2v_model.pkl` сервис работает на табличных + image модальностях (text_signal = 0).

## Запуск

### Docker (рекомендуется)

```bash
docker compose up --build
```

Поднимает два контейнера: `counterfeit-api` (порт 8000) и `postgres` (порт 5432).

### Локально

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Требуется запущенный PostgreSQL. Задать переменную окружения:
```
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/counterfeit
```

Открыть: http://localhost:8000  
Swagger UI: http://localhost:8000/docs

## API

### `POST /predict` — multipart/form-data

| Поле | Тип | Описание |
|------|-----|----------|
| image | file | Фото товара |
| name | string | Название товара |
| description | string | Описание |
| brand | string | Бренд |
| category | string | CommercialTypeName4 |
| price | float | Цена |
| item_time_alive | float | Дней на площадке |
| item_count_sales30 | float | Продажи за 30 дней |
| item_count_returns30 | float | Возвраты за 30 дней |
| seller_time_alive | float | Возраст продавца (дни) |
| seller_id | string | ID продавца (необязательно) |

Ответ:
```json
{
  "is_counterfeit": true,
  "probability": 0.87,
  "signals": {
    "multimodal_score": 0.87,
    "image_signal": 0.91,
    "text_signal": 0.43
  }
}
```

### `GET /predictions` — история предсказаний

Параметры: `is_counterfeit`, `seller_id`, `limit`, `offset`

### `POST /predictions/{id}/feedback` — оценка модератора

```json
{
  "correct": true,
  "true_label": true,
  "moderator_comment": "Явный контрафакт",
  "moderator_id": "moderator_01"
}
```

### `GET /seller-profiles/{seller_id}` — агрегат по продавцу

