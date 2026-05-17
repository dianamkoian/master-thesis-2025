# Подключение новой модели к сервису

Сервис спроектирован как pluggable: любая модель из исследовательской части
работы (M2-FE+, M2 mainline, ансамбли, R@P-optimal v2 и т. п.) может быть
подключена без изменений `main.py` или `worker.py`. Контракт описан в
`app/predictor_base.py` через абстрактный класс `BasePredictor`.

## Как переключить уже зарегистрированную модель

Достаточно выставить переменную окружения `PREDICTOR_TYPE`:

```bash
# Текущий deployed baseline (Doc2Vec + CLIP + CatBoost):
PREDICTOR_TYPE=d2v_catboost docker compose up

# Headline-конфигурация М2-FE+ (когда будет реализована):
PREDICTOR_TYPE=m2_fe_plus docker compose up
```

Доступные ключи смотри в словаре `PREDICTOR_REGISTRY` в `app/predictor.py`.
Значение по умолчанию — `d2v_catboost` (текущий baseline соревнования
Ozon eCup 2025; в финальной headline-конфигурации ВКР deprecated, см.
§ 3.4.3.3 и § 3.4.8 Negative Transfer 1).

## Как подключить новую модель

### Шаг 1. Реализовать `BasePredictor`

Создать файл `app/predictor_<name>.py` со своим подклассом. Минимальный
шаблон:

```python
from app.predictor_base import BasePredictor

class M2FeaturePlusPredictor(BasePredictor):
    """M2-FE+ из § 3.4.3.3: 38 tab + 512 CLIP + 50 TF-IDF SVD + 36 FE-фич."""

    name = "m2_fe_plus"
    description = "Headline индивидуального протокола: PR=0,7399 R@P=0,2213"

    def load(self) -> None:
        # 1) catboost_model_fe_plus.cbm
        # 2) tfidf_vectorizer.pkl + svd_50.pkl (текстовая ветка)
        # 3) brand_stats.parquet (groupby агрегаты на train)
        # 4) clip_centroids.npy + clip_kmeans_k8.pkl (CLIP-derived)
        # 5) img_scaler.pkl + img_pca.pkl (если используется)
        # Все артефакты должны лежать в ARTIFACTS_DIR (env or default).
        ...

    def predict(self, image_bytes, name, description, brand, tab_inputs):
        # 1) построить tab-row по feature_cols модели
        # 2) посчитать text-эмбеддинг через self.tfidf.transform → self.svd.transform
        # 3) посчитать CLIP-эмбеддинг и CLIP-derived structural (distances, cluster_id)
        # 4) собрать interactions (price_per_age и т. п.) и brand_agg (merge по brand_key)
        # 5) подать в self.model.predict_proba; вернуть словарь по контракту
        return {
            "is_counterfeit": bool(prob >= THRESHOLD),
            "probability": float(prob),
            "signals": {
                "multimodal_score": float(prob),
                "image_signal": float(image_only_prob),
                "text_signal": float(text_only_prob),
            },
        }
```

Контракт возвращаемого словаря — см. `app/predictor_base.py` (модульный
docstring). Поля `signals` обязательны для совместимости с фронтендом, но
если ваша модель не различает модальные сигналы, повторите там
`multimodal_score`.

### Шаг 2. Зарегистрировать в `PREDICTOR_REGISTRY`

В `app/predictor.py` добавить запись:

```python
from app.predictor_m2_fe_plus import M2FeaturePlusPredictor  # пример

PREDICTOR_REGISTRY = {
    "d2v_catboost": D2VCatBoostPredictor,
    "m2_fe_plus": M2FeaturePlusPredictor,           # новая запись
}
```

### Шаг 3. Положить артефакты в `artifacts/`

Каждая реализация сама знает, какие файлы ей нужны. Хорошая практика —
завести подпапку `artifacts/<predictor_name>/` и грузить относительно неё:

```
counterfeit_service/artifacts/
├── catboost_model.cbm           # d2v_catboost (легаси)
├── d2v_model.pkl
├── img_scaler.pkl
├── feature_cols.pkl
├── cat_cols.pkl
└── m2_fe_plus/                  # новая модель
    ├── catboost_model.cbm
    ├── tfidf.pkl
    ├── svd_50.pkl
    ├── brand_stats.parquet
    ├── clip_centroids.npy
    └── clip_kmeans_k8.pkl
```

### Шаг 4. Запустить с новой моделью

```bash
PREDICTOR_TYPE=m2_fe_plus docker compose up
```

В логах `/health` появится секция `predictor: m2_fe_plus` с описанием.

## Reasoning pipeline (§ 3.4.9.7)

Ключ `reasoning_pipeline` — wrapper, добавляющий канал verdict-conditioned
reasoning поверх любого зарегистрированного primary predictor'а. Реализован
в `app/predictor_with_reasoning.py`. Поддерживает два провайдера объяснений
с единым API-контрактом:

- **Rule-based template** (default, `app/template_reasoner.py`) —
  детерминированный генератор на правилах-эвристиках (SHAP top-15 § 3.4.7).
  Латентность < 1 мс, не зависит от GPU/MPS. Используется по умолчанию.
- **LLM** (`app/llm_explainer.py`, Qwen2.5-1.5B-Instruct, активация через
  `USE_LLM_REASONING=1`) — для деплоев с MPS/CUDA; на CPU латентность 15–30 с.

Архитектурный мотив. Эмпирически (§ 3.4.9.4) Qwen2.5-1.5B как Stage 2
вердикт-моделью даёт отрицательный перенос по `proba` (ROC ≈ 0,52); тем не
менее, текстовый reasoning сохраняет качество. В production-конфигурации
по умолчанию используется rule-based explainer ради детерминированности и
аудитопригодности (важно для регуляторного контекста); LLM-канал доступен
для деплоев с GPU/MPS как opt-in.

Контракт расширен двумя полями (обратно-совместимо):

```json
{
  "is_counterfeit": true,
  "probability": 0.62,
  "signals": { "multimodal_score": 0.62, "image_signal": 0.55, "text_signal": 0.71 },
  "reasoning": "Бренд Nespresso в названии в сочетании с указанием Xiaomi ...",
  "reasoning_mode": "borderline_explanation"
}
```

`reasoning` непустой только для borderline-объектов
(по умолчанию `proba ∈ (0.25, 0.75)`); для confident объектов поле = `null`,
`reasoning_mode` ∈ `{confident_negative, confident_positive, llm_error}`.

Активация:

```bash
PREDICTOR_TYPE=reasoning_pipeline docker compose up
# опционально: выбрать inner predictor (по умолчанию d2v_catboost)
PREDICTOR_TYPE=reasoning_pipeline PREDICTOR_INNER=d2v_catboost docker compose up
# опционально: подстроить пороги borderline-зоны
REASONING_BORDERLINE_LO=0.30 REASONING_BORDERLINE_HI=0.70 \
  PREDICTOR_TYPE=reasoning_pipeline docker compose up
# опционально: модель и устройство LLM (default: Qwen/Qwen2.5-1.5B-Instruct, auto)
LLM_REASONING_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  LLM_REASONING_DEVICE=mps \
  LLM_REASONING_MAX_TOKENS=200 \
  PREDICTOR_TYPE=reasoning_pipeline docker compose up
```

Требует `transformers` и `torch` (см. `requirements.txt`). Веса
Qwen2.5-1.5B-Instruct (~3 ГБ) скачиваются один раз в HuggingFace cache при
первом запросе к borderline-объекту. На M4 Pro с MPS — ≈ 5 секунд на объект.

### Persistence: интеграция с CRM через `predictions_async`

LLM-канал сохраняет `reasoning` и `reasoning_mode` в таблицу
`predictions_async` (см. `app/db/models.py`). Это позволяет внешним
CRM-системам (Bitrix24, AmoCRM, custom-back-office) подхватывать готовые
тексты прямо из БД, без обращения к ML-сервису:

```sql
-- Выгрузка автоблокировок за сутки для рассылки селлерам:
SELECT task_id, probability, reasoning, created_at
FROM predictions_async
WHERE reasoning_mode = 'blocking_explanation'
  AND created_at >= NOW() - INTERVAL '1 day';

-- Очередь ручной модерации (borderline-зона):
SELECT task_id, probability, reasoning, created_at
FROM predictions_async
WHERE reasoning_mode = 'borderline_explanation'
  AND status = 'done'
ORDER BY created_at DESC;
```

Индекс `idx_predictions_async_reasoning_mode` создаётся миграцией
`migrations/001_add_reasoning_columns.sql`; для свежей БД колонки и индекс
поднимаются автоматически из ORM-метаданных при первом старте.

## Текущее состояние

| Ключ                            | Класс                       | Файл                                  | Статус        |
|---------------------------------|-----------------------------|---------------------------------------|---------------|
| `d2v_catboost`                  | `D2VCatBoostPredictor`      | `app/predictor.py`                    | Реализован    |
| `reasoning_pipeline`            | `ReasoningPredictor`        | `app/predictor_with_reasoning.py`     | Реализован    |
| `m2_fe_plus`                    | `M2FeaturePlusPredictor`    | (не реализован)                       | TODO          |
| `m2_fe_plus_mainline_ensemble`  | `EnsemblePredictor`         | (не реализован)                       | TODO          |
| `r_at_p_optimal_v2`             | `EnsemblePredictor`         | (не реализован)                       | TODO          |

Чтобы реализовать TODO-предикторы, нужно сохранить артефакты из
соответствующих обучающих скриптов в `Диана_ВКР_финал/scripts/`
(например, `m2_fe_plus_individual_v2.py` сохраняет proba и `feature_names`,
но не CatBoost-модель и не FE-pipeline целиком — это нужно дописать перед
интеграцией в сервис). После реализации `m2_fe_plus` `reasoning_pipeline`
автоматически сможет использовать его как inner predictor через
`PREDICTOR_INNER=m2_fe_plus`.
