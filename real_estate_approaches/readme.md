# ВКР Дианы Мкоян — финальная версия

Магистерская диссертация на тему: **«Система контроля качества для e-commerce на основе кросс-доменной адаптации методов обнаружения фейков с использованием больших мультимодальных моделей»**.

НИУ ВШЭ, Факультет компьютерных наук, программа «Магистр по наукам о данных», 2026 год.
Научный руководитель — К. В. Быков.

Эта папка содержит **только индивидуальный вклад автора и связанные артефакты**. Все командные исходники, чужие файлы и устаревшие версии остались в основной директории проекта `master-thesis-2025/` и не вошли в данную сборку.

## Структура

```
Диана_ВКР_финал/
├── README.md                                      # этот файл
├── diploma/                                       # тексты диплома
│   ├── Диплом_Мкоян_Диана_final.docx             # финальный текст ВКР (источник истины)
│   ├── Титульный_лист_Мкоян_2026.docx             # заполненный титульный лист
│   ├── hse_reference.docx                         # стилевой шаблон ВШЭ для pandoc
│   └── Project_Passport.md                        # паспорт проекта (контекст)
│
├── figures/                                       # все 22 рисунка диплома
│   ├── рис_4_5_m2_fusion.png                      # архитектура M2 Multimodal Feature Fusion
│   ├── рис_4_6_m3_dual_level.png                  # M3 Dual-Level Fusion
│   ├── рис_4_7_m5_oof_stacking.png                # M5 OOF Stacking
│   ├── рис_4_8_cascade.png                        # каскад с LMM (главная схема § 3.4.9)
│   ├── рис_4_9_negative_results.png               # сравнение negative results (bar chart)
│   ├── рис_6_1_service_arch.png                   # production-сервис
│   ├── рис_7_1_rabbitmq_flow.png                  # RabbitMQ flow
│   ├── eda/                                       # 11 figures из 01_eda_defense.ipynb (Глава 2)
│   └── experiments/                               # 4 диагностики экспериментов (§ 3.4)
│
├── notebooks/                                     # 5 рабочих ноутов
│   ├── 01_eda_defense.ipynb                       # компактный EDA с 11 figures для слайдов
│   ├── 02_reproduction_fixed.ipynb                # адаптация 5 RE-методов M1–M5 (§ 3.4.2–4.4.6)
│   ├── 03_original_hybrid_cascade.ipynb           # каскад с CLIP zero-shot + conformal (§ 3.4.9.1, 4.4.9.2, 4.4.9.5)
│   ├── 04_lmm_reasoning.ipynb                     # каскад с Qwen2.5-1.5B zero-shot (§ 3.4.9.3)
│   ├── 05_qwen2vl_multimodal.ipynb                # каскад с Qwen2-VL-2B (готов к запуску)
│   ├── artifacts/                                 # CSV / PNG из ноута 02 (M1–M5)
│   ├── artifacts_original/                        # CSV / PNG / probas из ноута 03
│   ├── artifacts_lmm/                             # CSV / PNG / probas из ноута 04
│   ├── artifacts_lmm_lora/                        # LoRA-адаптер Qwen2.5 + scores (§ 3.4.9.4)
│   │   └── qwen_lora_adapter/                     # веса LoRA r=8, target=q_proj/v_proj
│   ├── test_proba_real_estate.npy                 # RAW probas для общекомандного стекинга
│   ├── val_proba_real_estate.npy                  # OOF val probas
│   ├── test_proba_no_te.npy / val_proba_no_te.npy # probas без target encoding (ablation, § 3.4.7)
│   ├── test_proba_tabnet.npy / val_proba_tabnet.npy   # TabNet probas (Negative transfer 4)
│   ├── test_proba_stack_cb_tabnet.npy             # ансамбль CB + TabNet (уступает CB-only)
│   └── feature_importance_team.csv                # топ-30 признаков CatBoost для § 3.4.11
│
├── scripts/                                       # python-скрипты
│   ├── team_split_v2_script.py                    # финальная мультимодальная модель на едином тесте
│   ├── tabnet_script.py                           # TabNet на едином сплите (§ 3.4.8)
│   └── tabpfn_v2_script.py                        # TabPFN v2 (не запущена, требует HF-токена)
│
├── counterfeit_service/                           # полный production-сервис с async-pipeline
│   ├── app/main.py                                # FastAPI: /predict (sync) + /predict-async + /result/{id}
│   ├── app/worker.py                              # aio_pika consumer (§ 5.4 Д. Мкоян)
│   ├── app/predictor.py                           # Feature Fusion CatBoost (§ 5.2 К. Азимова)
│   ├── app/schemas.py                             # Pydantic models
│   ├── app/db/                                    # async PostgreSQL (session/models/crud)
│   ├── static/                                    # веб-интерфейс (§ 5.5 С. Красовская)
│   ├── Dockerfile / Dockerfile.worker             # контейнеры для API и worker
│   ├── docker-compose.yml                         # api + worker + rabbitmq + postgres
│   ├── requirements.txt
│   ├── save_d2v_model.py                          # утилита для сериализации Doc2Vec из train
│   └── README.md                                  # архитектура + инструкция запуска
│
└── literature/                                    # литература по индивидуальному домену
    ├── Обзор_источников.md                        # сводный обзор 15 источников
    └── *.pdf                                      # PDF-источников [1]–[5]
```

## Воспроизведение

### Окружение

- Python 3.13
- Apple M4 Pro 24 ГБ RAM (MPS-ускорение для нейросетевых моделей)
- Стек: `pandas`, `numpy`, `scikit-learn` 1.7, `catboost` 1.2, `transformers` 5.x,
  `sentence-transformers`, `torch` 2.x с MPS, `pytorch_tabnet`, `peft`, `shap` 0.5,
  `matplotlib`, `seaborn`, `FastAPI`, `pika` / `aio_pika`, `Docker`

### Запуск ноутов

Перед запуском ноутов 02–05 необходимо иметь:

1. `ml_ozon_ounterfeit_train.csv` — основной датасет Ozon eCup 2025 (197 198 объектов), путь в ноутах: `Diana's folder/ml_ozon_ounterfeit_train.csv`
2. `clip_embeddings.parquet` — предвычисленные CLIP ViT-B/32 эмбеддинги (187 604 объекта × 512-dim), путь в ноутах: `counterfeit_service/clip_embeddings.parquet`
3. Для ноута 05 (Qwen2-VL): папка `data/ml_ozon_сounterfeit_train_images/` с raw-изображениями товаров (имена файлов = ItemID)

Эти большие файлы **не входят** в данную сборку — их публикация ограничена правилами соревнования. Источники указаны в [data_pointers.md](data_pointers.md).

### Финальный текст ВКР

Источник истины — готовый Word-документ `diploma/Диплом_Мкоян_Диана_final.docx`.
Ранние markdown-версии и pandoc-сборки сохранены в `diploma/_archive_old_versions/`.

Затем в Word: вклеить содержимое `Титульный_лист_Мкоян_2026.docx` в начало, добавить Содержание (References → Table of Contents), включить нумерацию страниц.

## Ключевые результаты

### Метрики на индивидуальном протоколе (test = 26 327 объектов)

| Модель | ROC-AUC | PR-AUC | R@P ≥ 0,9 |
|---|---|---|---|
| **M2 Multimodal CatBoost** | **0,9612** | **0,7228** | 0,1600 |
| M5 OOF Stacking | 0,9561 | 0,7167 | **0,2109** |
| M4 FADAML | 0,9595 | 0,7160 | 0,1392 |
| M3 Dual-Level Fusion | 0,9582 | 0,6800 | 0,0006 |
| M1 K-means + RF | 0,9566 | 0,6462 | 0,0012 |

### Метрики на едином командном протоколе (test = 58 410 объектов)

| Метрика | Значение | 95 % CI |
|---|---|---|
| ROC-AUC | 0,9548 | [0,9517; 0,9577] |
| PR-AUC | 0,7111 | [0,697; 0,725] |
| R@P ≥ 0,9 (raw) | 0,0222 | (хрупкая метрика, см. § 4.1) |

### Зафиксированные научные результаты

1. **Главный методологический результат** (§ 4.1): эмпирическое обоснование хрупкости метрики Recall@P ≥ 0,9 при сильном дисбалансе классов и протокольная норма для отчётности результатов
2. **Три negative transfer** (§ 3.4.8): SAFE-кросс-модальное несоответствие, Doc2Vec-семантические энкодеры, TabNet-нейросетевая альтернатива CatBoost
3. **Два значимых negative result для foundation-моделей** в zero-shot режиме: CLIP (§ 3.4.9.2, ROC = 0,48) и Qwen2.5-1.5B (§ 3.4.9.3, ROC = 0,51); + контрольный supervised-эксперимент с LoRA (§ 3.4.9.4) подтверждает фундаментальное ограничение модели данного размера
4. **Ablation на target encoding** (§ 3.4.7): неожиданный 9-кратный рост R@P ≥ 0,9 при удалении LOO-target признаков на едином командном тесте
5. **Оригинальный вклад**: гибридная каскадная архитектура с LLM-reasoning + conformal prediction (§ 3.4.9), готовая инфраструктура для замены Stage 2 на более крупную модель

## Сводка вклада автора

| Раздел | Тип | Описание |
|---|---|---|
| § 1.5.4 (с подразделами 1.5.4.1–1.5.4.6) | **Индивидуальный** | Самостоятельный обзор 5 источников по детекции мошеннических объявлений в недвижимости + 12 источников по визуально-языковым LMM 2024–2025 ([54]–[65]) |
| § 3.4 (с § 3.4.1–3.4.12) | **Индивидуальный** | Адаптация M1–M5, ablation на target encoding, TabNet NT4, каскадная архитектура с CLIP/Qwen/LoRA, conformal predictor |
| § 4.1 | Совместно с С. Красовской | Формализация методологического результата о хрупкости R@P ≥ 0,9 |
| § 5.4 (с § 5.4.1–5.4.3) | **Индивидуальный** | Подсистема асинхронной обработки на RabbitMQ |
| § 5.6 (с § 5.6.1–5.6.4) | **Индивидуальный** | Функциональное и нагрузочное тестирование, контроль воспроизводимости |
| Главы 1–5 | Текст индивидуально, методология совместно | Согласование структуры и методологии совместно с участниками группы; весь текст в настоящей рукописи написан автором |
