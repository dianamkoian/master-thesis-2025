# Адаптация методов fraud detection из финтеха для детекции контрафакта на Ozon

Задача детекции контрафактных товаров на маркетплейсе структурно близка к задаче обнаружения мошеннических банковских транзакций: в обоих случаях ищут редкие неаутентичные события среди потока легитимных, при сильном дисбалансе классов и требовании высокой точности (low false-positive rate). В данной работе мы переносим набор практических приёмов и методологических решений из литературы по credit-card fraud detection на нашу задачу и проверяем три гипотезы об источнике потенциального улучшения.

> Перенос осуществляется на уровне практических приёмов и методологии, а не на уровне моделей или данных. В узком формальном смысле (DANN, CORAL, MMD) кросс-доменная адаптация не применялась.

## Метрики

- **Recall@P≥0.9** (основная) — доля выявленного контрафакта на пороге, при котором precision ≥ 0.9. Соответствует операционному режиму маркетплейса.
- **PR-AUC** (вспомогательная) — обоснование выбора см. в [11].
- **ROC-AUC** — справочно, не используется для принятия решений из-за дисбаланса классов.

Все метрики считаются на едином seller-based split (canonical train = 135626, test зафиксирован).

## Baseline (E0)

CatBoost на 51 табличном признаке (`fintech_experiment.ipynb`, Cell 4). Эталон для всех последующих экспериментов:

| Метрика | E0_canon |
|---|---|
| ROC-AUC | 0.9208 |
| PR-AUC | 0.6587 |
| Recall@P≥0.9 | 0.1721 |

## Гипотеза H1 — Decision Rule (E1, E6)

**Вопрос:** можно ли улучшить Recall@P≥0.9, не меняя признаки, только сдвигая порог или вес позитивного класса?

| Эксперимент | Метод | Источник | PR-AUC | R@P≥0.9 |
|---|---|---|---|---|
| E1 | Threshold optimization на validation | [11] | 0.6421 | 0.2196* |
| E6 spw=15 | Scale pos weight = 15 | [6] | 0.6541 | 0.2078 |
| E6 spw=25 | spw = 25 | [6] | 0.6529 | 0.1941 |
| E6 spw=35 | spw = 35 | [6] | 0.6507 | 0.1794 |
| E6 spw=50 | spw = 50 | [6] | 0.6338 | 0.0985 |

*E1 обучен на E0_old (train=110970); oracle-метрика идентична E0_old by construction.

**Вывод:** threshold не является bottleneck, class weighting даёт лишь маргинальный эффект, при росте spw метрика последовательно деградирует. **H1 не подтверждается.**

## Гипотеза H2 — Feature engineering из финтеха (E2, E3, E4a)

**Вопрос:** можно ли улучшить Recall@P≥0.9 обработкой существующих признаков без добавления новых модальностей?

| Эксперимент | Метод | Источник | PR-AUC | R@P≥0.9 |
|---|---|---|---|---|
| E2 | RandomUnderSampler + Dal Pozzolo calibration | [4] | 0.6480 | 0.1725 |
| E3 | +3 deviation features из seller train stats | [2], [12] | 0.6530 | 0.1225 |
| E4a | TF-IDF (300 feat, ngram=(1,2)) вместо встроенного кодирования текста | [3], [9] | 0.4535 | 0.0000 |

**Вывод:** E2 и E3 поднимают PR-AUC на 0.003–0.007, но Recall@P≥0.9 не сдвигается или падает; E4a — явная деградация (negative transfer). **H2 не подтверждается:** прирост PR-AUC локализован в зоне низкой/средней precision и не транслируется в целевую зону.

## Гипотеза H3 — Multimodal extension (E4b, E5, E7)

**Вопрос:** требуется ли добавление новой модальности для сдвига Recall@P≥0.9?

| Эксперимент | Метод | Источник | PR-AUC | R@P≥0.9 |
|---|---|---|---|---|
| E4b | CLIP-512 naive (без PCA) | [10] | 0.5825 | 0.0206 |
| **E5** | **CLIP-512 → PCA-25 (57.9% var) + tabular** | **[10], [5], [7]** | **0.6724** | **0.2956** |
| **E7** | **e5-small text emb → PCA-25 + tabular** | стандартная техника | **0.6660** | **0.3299** |

E4b показывает, что прямая конкатенация 512-мерных CLIP-эмбеддингов с табличными признаками приводит к переобучению (best_iteration = 7 из 500): согласно Grinsztajn et al. [5], GBDT деградирует при подаче высокоразмерных плотных признаков с низкой индивидуальной информативностью. Это инженерная проблема размерности, а не negative transfer в смысле [13]. PCA-сжатие до 25 компонент решает проблему: E5 и E7 — единственные эксперименты, значимо превзошедшие E0_canon одновременно по обеим метрикам.

**H3 подтверждается:** новая модальность (визуальная в E5, текстовая в E7) — необходимое условие сдвига Recall@P≥0.9 на seller-based split.

## Главный результат и статистическая значимость

Bootstrap-CI (paired test set, n=1000 итераций):

| Сравнение | Δ R@P≥0.9 | 95% CI | Significant |
|---|---|---|---|
| E5 vs E0_canon | +0.1235 | [+0.007, +0.203] | ✅ |
| E7 vs E0_canon | +0.1332 | [+0.036, +0.239] | ✅ |
| E7 vs E5 | +0.0308 | [−0.014, +0.075] | ❌ (не различимы) |
| E5 vs E6 spw=15 | +0.0767 | [−0.004, +0.143] | ❌ (изолирует вклад CLIP, не значим) |

E5 и E7 значимо лучше baseline. Между собой E5 и E7 статистически неразличимы — что согласуется с гипотезой об эквивалентности визуальной и текстовой модальности на этой задаче (товарные карточки на Ozon сильно текстовые, изображения часто похожи у оригинала и контрафакта).

## Secondary baselines

Чтобы исключить, что улучшение E5/E7 объясняется силой самого CatBoost, обучены две альтернативные tabular-only модели (см. `secondary_baselines.csv`):

| Модель | Признаки | PR-AUC | R@P≥0.9 |
|---|---|---|---|
| LightGBM | numeric + SellerID | 0.1967 | 0.0005 |
| LogReg + StandardScaler | numeric only | 0.2117 | 0.0000 |
| **CatBoost E0_canon (reference)** | all (incl. text + SellerID) | **0.6587** | **0.1721** |

CatBoost существенно превосходит обе альтернативы при использовании всех признаков — этим обусловлен выбор его в качестве основной архитектуры; результаты E5/E7 не сводятся к простой замене tabular-модели.

## Проверка leakage (random-split sanity)

В дополнение к seller-based split проведена проверка на случайном разбиении (`random_split_check.csv`). На random split метрика E0 искусственно завышается:

- random split: R@P≥0.9 = **0.6307**
- seller-based split: R@P≥0.9 = **0.1721**
- разница: **в 3.7 раза**

Исключение `SellerID` из признаков закрывает лишь 11% разрыва (0.6307 → 0.5828). Оставшиеся 89% утечки проходят через косвенные каналы: seller aggregates, текстовые категориальные признаки, числовые паттерны продавца. Это подтверждает, что seller-based split — единственный методологически валидный сценарий оценки, и обосновывает выбор E0_canon как эталона для всех последующих экспериментов.

## Структура папки

```
fintech_approaches/
├── README.md                       # этот файл
├── fintech_experiment.ipynb        # основной ноутбук, E0–E6
├── E7_colab_runner.ipynb           # E7 (e5-small embeddings, Colab)
├── final_results.md                # итоговая текстовая сводка
├── results_table_v2.csv            # сводная таблица метрик E0–E7
├── bootstrap_ci.csv                # bootstrap-CI каждого эксперимента (n=1000)
├── bootstrap_pairwise.csv          # paired bootstrap-тесты основных сравнений
├── secondary_baselines.csv         # LightGBM, LogReg для сверки с CatBoost
├── random_split_check.csv          # проверка корректности seller-based split
├── test_proba_*.npy                # предсказанные вероятности на test (15 моделей)
├── y_test_canon.npy                # ground truth для canonical split
├── y_test_random_split.npy         # ground truth для random split (sanity)
├── pr_curve_*.png                  # PR-curves для основных экспериментов
├── pr_curves_*comparison.png       # сравнительные PR-curves
├── threshold_search_*.png          # визуализация поиска порога (E1, E2, E3)
└── catboost_info/                  # логи обучения CatBoost
```

Эмбеддинги CLIP и e5-small лежат вне этой папки (в `data/` или генерируются в Colab из исходных изображений/текстов).

## Сводная таблица

| Блок | Эксперимент | PR-AUC | R@P≥0.9 | Δ R@P≥0.9 vs E0 |
|---|---|---|---|---|
| Baseline | E0_canon | 0.6587 | 0.1721 | 0.0 |
| H1 | E1 | 0.6421 | 0.2196* | +0.0475 |
| H1 | E6 spw=15 | 0.6541 | 0.2078 | +0.0357 |
| H1 | E6 spw=25 | 0.6529 | 0.1941 | +0.0220 |
| H1 | E6 spw=35 | 0.6507 | 0.1794 | +0.0073 |
| H1 | E6 spw=50 | 0.6338 | 0.0985 | −0.0736 |
| H2 | E2 | 0.6480 | 0.1725 | +0.0004 |
| H2 | E3 | 0.6530 | 0.1225 | −0.0496 |
| H2 | E4a TF-IDF | 0.4535 | 0.0000 | −0.1721 |
| H3 | E4b CLIP naive | 0.5825 | 0.0206 | −0.1515 |
| **H3** | **E5 CLIP + PCA-25** | **0.6724** | **0.2956** | **+0.1235** |
| **H3** | **E7 e5-small + PCA-25** | **0.6660** | **0.3299** | **+0.1578** |

*обучен на E0_old train, не строго сопоставим.

## Литература

[1] Amin M. H. M., Hassan H. H., Mohd Sani N. S., Nasruddin Z. A. Class-Weighted Dempster–Shafer in Dual-Level Fusion for Multimodal Fake Real Estate Listings Detection. PeerJ Computer Science. 2025. DOI: 10.7717/peerj-cs.2797.
[2] Bahnsen A. C., Aouada D., Stojanovic A., Ottersten B. Feature Engineering Strategies for Credit Card Fraud Detection. Expert Systems with Applications. 2016.
[3] Boulieris P., Pavlopoulos J., Xenos A., Vassalos V. Fraud Detection with Natural Language Processing. Machine Learning. 2024. Vol. 113, No. 8. P. 5087–5108. DOI: 10.1007/s10994-023-06354-5.
[4] Dal Pozzolo A., Caelen O., Johnson R. A., Bontempi G. Calibrating Probability with Undersampling for Unbalanced Classification. IEEE Symposium Series on Computational Intelligence (SSCI). 2015.
[5] Grinsztajn L., Oyallon E., Varoquaux G. Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data? NeurIPS. 2022.
[6] He H., Garcia E. A. Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering. 2009. Vol. 21, No. 9.
[7] Jolliffe I. T. Principal Component Analysis (2nd ed.). Springer. 2002.
[8] Mutemi A., Bacao F. A Numeric-Based Machine Learning Design for Detecting Organized Retail Fraud in Digital Marketplaces. Scientific Reports. 2023. Vol. 13, Art. 12499. DOI: 10.1038/s41598-023-38304-5.
[9] Prokhorenkova L., Gusev G., Vorobev A., Dorogush A. V., Gulin A. CatBoost: Unbiased Boosting with Categorical Features. NeurIPS. 2018.
[10] Radford A. et al. Learning Transferable Visual Models From Natural Language Supervision. ICML. 2021.
[11] Saito T., Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE. 2015. Vol. 10, No. 3.
[12] Cao B., Mao M., Viidu S., Yu P. Collective Fraud Detection Capturing Inter-Transaction Dependency. Proceedings of the KDD 2017 Workshop on Anomaly Detection in Finance. PMLR 71. 2018. P. 66–75.
[13] Zhang W., Deng L., Zhang L., Wu D. A Survey on Negative Transfer. IEEE/CAA Journal of Automatica Sinica. 2023. Vol. 10, No. 2. P. 305–329. DOI: 10.1109/JAS.2022.106004.
