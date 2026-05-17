# Большие модели — скачивание

Бинарные файлы моделей размером более ~5 МБ не хранятся в git-репозитории
(ограничение GitHub — 100 МБ на файл, плюс репозиторий не должен раздуваться), поэтому мы перенесли их на [диск](https://drive.google.com/drive/folders/1FCPi6nSqWrSHM1um87Mp92PrQt-3ocnb?usp=share_link).
Их нужно скачать отдельно и положить по указанным путям перед запуском сервиса.

Мелкие артефакты (метаданные, скейлеры, PCA, fold-модели CatBoost, lookup-таблицы)
лежат в репозитории и доступны сразу.

## Файлы для скачивания

| Путь в проекте | Размер | Назначение |
|----------------|--------|-----------|
| `artifacts/d2v_model.pkl` | ~437 МБ | Doc2Vec-модель для baseline `D2VCatBoostPredictor` | 
| `artifacts/cdsm_v3/amm_thinker.pkl` | ~136 МБ | AMM MMD-Thinker v4 (канал p_AMM) |
| `artifacts/cdsm_v3/ftmff_catboost.cbm` | ~23 МБ | FT-MFF Fusion CatBoost (канал p_FT-MFF) |
| `artifacts/cdsm_v3/rmm_catboost.cbm` | ~5 МБ | RMM CatBoost (канал p_RMM) |
| `artifacts/cdsm_v3/rmm_catboost_typosquat.cbm` | ~5 МБ | RMM CatBoost (typosquat-вариант) | 

## Опционально

| Путь в проекте | Размер | Назначение |
|----------------|--------|-----------|
| `artifacts/d2v_model.pkl.legacy_backup` | ~529 МБ | Резервная копия предыдущей версии Doc2Vec; для работы сервиса не требуется | 

## Как использовать

1. Скачать файлы по ссылкам выше.
2. Положить каждый файл по указанному пути внутри `counterfeit_service/`.
3. Запустить сервис согласно `README.md` / `QUICKSTART.md`.

Без этих файлов сервис запускается в ограниченном режиме: предиктор
`CDSMV3HeadlinePredictor` работает на frozen canonical probas из lookup-таблиц,
а полный live-инференс по тяжёлым моделям недоступен.
