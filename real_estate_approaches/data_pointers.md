# Внешние данные

В настоящей сборке намеренно не публикуются большие исходные данные. Ниже указано, где их получить и в какой путь поместить для воспроизведения экспериментов.

## 1. Основной датасет (обязательно)

**Файл:** `ml_ozon_ounterfeit_train.csv`
**Содержимое:** 197 198 товарных карточек с разметкой `resolution ∈ {0, 1}`, 6,62 % контрафакта, 45 признаков.
**Источник:** соревнование Ozon eCup 2025, трек «Контроль качества» (Kaggle / Ozon Platform).
**Путь:** `data/ml_ozon_ounterfeit_train.csv` (≈ 191 МБ).

**Файл:** `ml_ozon_ounterfeit_test.csv`
**Содержимое:** 22 760 товарных карточек без меток (для отправки на соревнование).
**Путь:** `data/ml_ozon_ounterfeit_test.csv` (≈ 26 МБ).

## 2. CLIP-эмбеддинги (обязательно для ноутов 02–05)

**Файл:** `clip_embeddings.parquet`
**Содержимое:** предвычисленные CLIP ViT-B/32 эмбеддинги (187 604 объекта × 512-dim) для 95,1 % обучающей выборки.
**Получение:** один раз вычисляется через CLIP ViT-B/32 на raw-изображениях; в репозиторий не публикуется ввиду размера.

## 3. Raw-изображения (обязательно только для ноута 05)

**Папка:** `data/ml_ozon_counterfeit_train_images/`
**Содержимое:** PNG-файлы товарных карточек, имена соответствуют `ItemID`.
**Объём:** около 25 ГБ; в репозитории не публикуются ввиду размера и правил Ozon eCup.

Без этой папки квантитативный эксперимент в `05_qwen2vl_multimodal.ipynb` не запустится — ноут потребует Qwen2-VL-2B-Instruct + raw images. Альтернатива — качественная демонстрация на нескольких объектах из публичной части тестовой выборки `data/ml_ozon_counterfeit_test_images/`.

## 4. Модельные веса Qwen2.5 / Qwen2-VL / Qwen2-VL-LoRA-adapter

- **Qwen2.5-1.5B-Instruct** — загружается автоматически HuggingFace Transformers по имени `Qwen/Qwen2.5-1.5B-Instruct` (требуется доступ к HF Hub).
- **Qwen2-VL-2B-Instruct** — `Qwen/Qwen2-VL-2B-Instruct`, ≈ 4,2 ГБ безопасно качается через `huggingface_hub.snapshot_download`.
- **LoRA-адаптер автора** (300 train borderline, r = 8, q_proj / v_proj) — приложен в `notebooks/artifacts_lmm_lora/qwen_lora_adapter/`, размер ≈ 4 МБ.

## 5. Сервис в production

Развёрнутая копия сервиса доступна по адресу `https://marketplace-fraud.ru` (Timeweb Cloud). Health-check: `GET /health`.

## 6. Опционально

- **`ml_ozon_counterfeit_test_images/`** — публичная часть тестовой выборки соревнования (22 735 PNG, ≈ 3,3 ГБ); используется только для качественной демонстрации Qwen2-VL.
- **Веса CatBoost-инференса production-сервиса** (`catboost_model.cbm`, `tfidf_vectorizer.pkl`, `img_scaler.pkl` и др.) — артефакты раздела команды, не относящиеся напрямую к диссертации автора; для полного воспроизведения см. README соответствующих репозиториев соавторов.

## 7. Командные вероятности соавторов (входы для стекинг-экспериментов)

Эксперименты по кросс-доменному стекингу (ноутбук `07_karina_channel_variants.ipynb`, скрипты в `scripts/sonya_pilot/`) используют как входные данные вероятности моделей соавторов на едином командном test-сплите (n = 58 410). Эти файлы — результаты работы других участников команды; в индивидуальную папку автора они не включены и предоставляются соответствующими соавторами.

| Файл | Источник | Содержание | Куда положить |
|------|----------|-----------|----------------|
| `test_proba_karina_team.npy` | К. Азимова (Apps & Ads) | вероятности доменной модели мобильных приложений | `notebooks/` |
| `test_proba_fusion_team.npy` | С. Красовская (FinTech) | финальная мультимодальная модель FinTech (headline) | `scripts/sonya_pilot/proba/` |
| `test_proba_e0_team_clean.npy` | С. Красовская (FinTech) | FinTech baseline E0 (чистый табличный, без текста) | `scripts/sonya_pilot/proba/` |
| `test_proba_e0_team_full.npy` | С. Красовская (FinTech) | FinTech baseline E0 (текст как category) | `scripts/sonya_pilot/proba/` |
| `test_proba_e5_team.npy` | С. Красовская (FinTech) | FinTech E5 (visual-only) | `scripts/sonya_pilot/proba/` |
| `test_proba_e7_team.npy` | С. Красовская (FinTech) | FinTech E7 (text-only) | `scripts/sonya_pilot/proba/` |

Собственные вероятности автора (`test_proba_diana_*`, `test_proba_real_estate*` и др.) и единый командный сплит (`notebooks/team_splits/`) остаются в репозитории.
