# Обнаружение контрафактных товаров на Ozon с использованием методов детекции фейков в социальных сетях

Задача обнаружения контрафакта на маркетплейсе структурно похожа на задачу детекции фейковых новостей в социальных сетях: в обоих случаях нужно найти неаутентичный контент среди огромного потока настоящего, используя текстовую и визуальную информацию. В данной работе мы адаптируем методы из трёх статей по детекции фейков в соцсетях и исследуем их применимость к задаче контрафакта.

## Модели

### 1. SpotFake-Ozon 

**Статья:** Singhal et al. *SpotFake: A Multi-modal Framework for Fake News Detection*. ACM MM, 2020.

**Идея:** каждая модальность обрабатывается независимым энкодером, затем представления конкатенируются и подаются в классификатор (late fusion).

**Архитектура:**

```
Текст (TEXT_DIM)  ──► TextEncoder  ──► text_emb (128)
                                                      \
Картинка (512)    ──► ImageEncoder ──► img_emb  (128) ──► Concat (384) ──► Classifier ──► sigmoid
                                                      /
Таблица (TAB_DIM) ──► TabEncoder   ──► tab_emb  (128)
```

Каждый `ModalityEncoder`: `Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm`

**Адаптации под нашу задачу:**

| Параметр | SpotFake (оригинал) | SpotFake-Ozon |
|---|---|---|
| Домен | Фейковые новости в соцсетях | Контрафакт на Ozon |
| Текстовый энкодер | BERT | ruBERT|
| Визуальный энкодер | VGG-19 (fine-tune) | CLIP embeddings (готовые, 512 dim) |
| Модальности | 2 (текст + картинка) | 3 (+ табличные признаки) |
| Fusion | Конкатенация | Конкатенация |
| Loss | BCE | Focal Loss (γ=2, α=0.75) |
| Дисбаланс | Не учитывается | WeightedRandomSampler |
| Оптимизатор | Adam | AdamW + CosineAnnealingLR |

**Ноутбук:** `spotfake_ozon_final.ipynb`


### 2. SAFE-Ozon в разработке

**Статья:** Zhou et al. *SAFE: Similarity-Aware Multi-modal Fake News Detection*. PAKDD, 2020. [arXiv:2003.04981](https://arxiv.org/abs/2003.04981)

**Идея:** контрафакт часто использует картинку оригинального товара с другим текстом, или наоборот — копирует описание оригинала, но показывает другой товар. SAFE предлагает явно измерять **семантическое несоответствие между текстом и картинкой** и использовать его как дополнительный сигнал.

**Что добавляем к SpotFake-Ozon:**

1. Получаем CLIP-эмбеддинг текста (через `openai/clip-vit-base-patch32`)
2. Считаем косинусное сходство между CLIP текста и CLIP картинки:
   ```python
   similarity = F.cosine_similarity(clip_text_emb, clip_img_emb)
   ```
3. Добавляем `similarity` как дополнительный признак в классификатор

**Гипотеза:** у контрафакта сходство между текстом и картинкой ниже, чем у оригинала.

**Почему это работает с нашими данными:** у нас уже есть CLIP-эмбеддинги картинок (`clip_embeddings.parquet`). Нужно только получить CLIP-эмбеддинги текстов — это один дополнительный шаг.

---

### 3. TGA-Ozon в разработке

**Статья:** Yang et al. *Multi-modal Transformer for Fake News Detection*. Mathematical Biosciences and Engineering, 2023. [doi:10.3934/mbe.2023657](https://doi.org/10.3934/mbe.2023657)

**Идея:** вместо простой конкатенации эмбеддингов (как в SpotFake) использовать **механизм attention** для динамического взвешивания модальностей. Это позволяет модели самой решать, какой модальности доверять больше для конкретного товара.

**Что меняем в SpotFake-Ozon:**

Заменяем блок `Concat → Classifier` на `CrossModalAttention → Classifier`:

```
text_emb (128)                    mean_pool (128)
img_emb  (128)  ──► MultiheadAttention ──►              ──► Concat (256) ──► Classifier
tab_emb  (128)      + residual + LayerNorm  max_pool (128)
```

**Адаптации от оригинального TGA:**

| Параметр | TGA (оригинал) | TGA-Ozon |
|---|---|---|
| Визуальный энкодер | ViT (обучается) | CLIP embeddings (готовые) |
| Данные | Weibo, Twitter | Ozon |
| Модальности | 2 (текст + картинка) | 3 (+ таблица) |

---

## План экспериментов

| Модель | Идея из статьи | Статус |
|---|---|---|
| SpotFake-Ozon | Late fusion, конкатенация | 
| SAFE-Ozon | + Cross-modal similarity | 
| TGA-Ozon | + CrossModalAttention | 

Каждая следующая модель добавляет одну идею из статьи — это позволяет измерить вклад каждого метода отдельно (incremental evaluation).

---

## Структура проекта

```
.
├── spotfake_ozon_final.ipynb   # SpotFake-Ozon
├── safe_ozon.ipynb             # SAFE-Ozon (в разработке)
├── tga_ozon.ipynb              # TGA-Ozon (в разработке)
├── ozon_train.csv              # обучающие данные
├── clip_embeddings.parquet     # CLIP-эмбеддинги картинок
```


## Литература

1. Singhal S. et al. **SpotFake: A Multi-modal Framework for Fake News Detection**. ACM MM, 2020.
2. Zhou X. et al. **SAFE: Similarity-Aware Multi-modal Fake News Detection**. PAKDD, 2020. arXiv:2003.04981
3. Yang P. et al. **Multi-modal Transformer for Fake News Detection**. Mathematical Biosciences and Engineering, 2023. doi:10.3934/mbe.2023657
