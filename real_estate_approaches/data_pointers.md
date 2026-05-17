# Внешние данные

В настоящей сборке намеренно не публикуются большие исходные данные. Ниже указано, где их получить и в какой путь поместить для воспроизведения экспериментов.

## 1. Основной датасет (обязательно)

**Файл:** `ml_ozon_ounterfeit_train.csv`
**Содержимое:** 197 198 товарных карточек с разметкой `resolution ∈ {0, 1}`, 6,62 % контрафакта, 45 признаков.
**Источник:** соревнование Ozon eCup 2025, трек «Контроль качества» (Kaggle / Ozon Platform).
**Путь:** `Diana's folder/ml_ozon_ounterfeit_train.csv` (≈ 191 МБ).

**Файл:** `ml_ozon_ounterfeit_test.csv`
**Содержимое:** 22 760 товарных карточек без меток (для отправки на соревнование).
**Путь:** `Diana's folder/ml_ozon_ounterfeit_test.csv` (≈ 26 МБ).

## 2. CLIP-эмбеддинги (обязательно для ноутов 02–05)

**Файл:** `clip_embeddings.parquet`
**Содержимое:** предвычисленные CLIP ViT-B/32 эмбеддинги (187 604 объекта × 512-dim) для 95,1 % обучающей выборки.
**Получение:** один раз вычисляется через CLIP ViT-B/32 на raw-изображениях; в репозитории команды артефакт расположен в `counterfeit_service/clip_embeddings.parquet`.

## 3. Raw-изображения (обязательно только для ноута 05)

**Папка:** `data/ml_ozon_сounterfeit_train_images/`
**Содержимое:** PNG-файлы товарных карточек, имена соответствуют `ItemID`.
**Объём:** около 25 ГБ; в репозитории не публикуются ввиду размера и правил Ozon eCup.

Без этой папки квантитативный эксперимент в `05_qwen2vl_multimodal.ipynb` не запустится — ноут потребует Qwen2-VL-2B-Instruct + raw images. Альтернатива — качественная демонстрация на нескольких объектах из публичной части тестовой выборки `data/ml_ozon_counterfeit_test_images/`.

## 4. Модельные веса Qwen2.5 / Qwen2-VL / Qwen2-VL-LoRA-adapter

- **Qwen2.5-1.5B-Instruct** — загружается автоматически HuggingFace Transformers по имени `Qwen/Qwen2.5-1.5B-Instruct` (требуется доступ к HF Hub).
- **Qwen2-VL-2B-Instruct** — `Qwen/Qwen2-VL-2B-Instruct`, ≈ 4,2 ГБ безопасно качается через `huggingface_hub.snapshot_download`.
- **LoRA-адаптер автора** (300 train borderline, r = 8, q_proj / v_proj) — приложен в `notebooks/artifacts_lmm_lora/qwen_lora_adapter/`, размер ≈ 4 МБ.

## 5. Сервис в production

Развёрнутая копия сервиса доступна по адресу `http://5.129.242.72:8000` (Timeweb Cloud). Health-check: `GET /health`.

## 6. Опционально

- **`ml_ozon_counterfeit_test_images/`** — публичная часть тестовой выборки соревнования (22 735 PNG, ≈ 3,3 ГБ); используется только для качественной демонстрации Qwen2-VL.
- **Веса CatBoost-инференса production-сервиса** (`catboost_model.cbm`, `tfidf_vectorizer.pkl`, `img_scaler.pkl` и др.) — артефакты раздела команды, не относящиеся напрямую к диссертации автора; для полного воспроизведения см. README соответствующих репозиториев соавторов.
