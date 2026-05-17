"""
HCDM live-inference helpers (§ 5.4.6 ВКР).

Дополняют `_cdsm_loaders` функциями для HCDM-специфичных каналов:
  - `predict_mode3_alone()` — извлекает p3 из AMM pickle без применения meta-LR
    (HCDM использует Mode 3 standalone, не Adaptive Thinking meta);
  - `predict_karina_image()` — LogReg на CLIP-эмбеддинге (Baseline C мобильного
    домена, § 4.1);
  - `compute_typosquat_features()` — три Deng-typosquatting признака
    [Deng et al., 2020] для расширения tab_inputs канала p_realestate
    (RMM-typosquat, § 5.4.6.9).

Композиция HCDM:
    p_HCDM(x) = 0,875 · p_social(x)
              + 0,075 · p_mobile_image(x)
              + 0,025 · p_realestate(x)
              + 0,025 · p_fintech(x)
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def _mode3_engineer_features(
    name: str, description: str, brand: str, tab_features: dict,
) -> Dict[str, float]:
    """Воспроизведение `engineer_features()` из тренировочного ноутбука Альбины
    `safe_ozon_v4_mmd__1_.ipynb`, cell 12. Создаёт 36 engineered признаков для
    Mode 3 на основе пользовательского ввода.

    Контракт missing-values: training-pipeline применял `engineer_features()`
    на DataFrame с NaN, а затем `train_df[cols_mode3].fillna(-1)` в cell 18.
    Мы воспроизводим эту семантику: для отсутствующих числовых входов
    возвращаем `np.nan`, который позже в сборке вектора Mode 3 заменяется
    на `-1.0` (см. `predict_mode3_alone`). Это отличает «пользователь
    оставил поле пустым» от «пользователь ввёл 0».
    """
    SUSPICIOUS_WORDS = [
        'реплика', 'копия', 'аналог', 'совместим', 'noname',
        'compatible', 'оригинал', 'original', 'brand new',
    ]
    out: Dict[str, float] = {}

    def _fnum(k):
        """Float с явным NaN для отсутствующего значения."""
        v = tab_features.get(k)
        if v in (None, ''):
            return float('nan')
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('nan')

    # 1. is_null_* флаги — однозначно бинарны (1 если значение реально пропущено)
    out['is_null_brand_name'] = float(not (brand or '').strip())
    out['is_null_description'] = float(not (description or '').strip())
    for col in ['GmvTotal7', 'GmvTotal30', 'GmvTotal90',
                'rating_5_count', 'ExemplarAcceptedCountTotal7']:
        # UI не передаёт эти поля → всегда is_null=1 для них.
        v = tab_features.get(col)
        out[f'is_null_{col}'] = float(v in (None, ''))

    # 2. Ratio features — NaN распространяется через арифметику, как в pandas.
    # На training: r=NaN → return_rate=NaN → cell18 fillna(-1) → -1.
    for w in [7, 30, 90]:
        s = _fnum(f'item_count_sales{w}')
        r = _fnum(f'item_count_returns{w}')
        f = _fnum(f'item_count_fake_returns{w}')
        out[f'return_rate{w}'] = r / (s + 1)             # NaN-propagating
        out[f'fake_return_rate{w}'] = f / (s + 1)
        out[f'fake_to_all_returns{w}'] = f / (r + 1)
        # zero_sales: NaN == 0 → False → 0 (match training: pandas (NaN==0).astype(float) = 0)
        out[f'zero_sales{w}'] = float(s == 0) if not np.isnan(s) else 0.0

    # 3. Log-трансформации: np.log1p(NaN) = NaN → fillna(-1) на сборке
    for col in ['item_time_alive', 'seller_time_alive', 'PriceDiscounted']:
        v = _fnum(col)
        out[f'log_{col}'] = float(np.log1p(v))           # NaN-propagating

    # 4. Interaction age (NaN * X = NaN)
    out['log_age_interaction'] = out['log_item_time_alive'] * out['log_seller_time_alive']

    # 5. Текстовые мета-признаки — текст всегда есть (даже пустая строка)
    desc_len = len(description or '')
    name_len = len(name or '')
    brand_len = len(brand or '')
    out['description_length'] = float(desc_len)
    out['name_length'] = float(name_len)
    out['brand_length'] = float(brand_len)
    out['has_description'] = float(bool((description or '').strip()))
    out['has_brand'] = float(bool((brand or '').strip()))
    out['text_complexity'] = desc_len / (name_len + 1)
    out['desc_name_ratio'] = float(np.log1p(desc_len)) / float(np.log1p(name_len + 1))

    # 6. Кросс-модальные сигналы — всегда определены (по тексту)
    brand_l = (brand or '').lower()
    name_l = (name or '').lower()
    out['brand_in_name'] = float(bool(brand_l) and brand_l in name_l)
    out['brand_name_conf'] = float(bool(brand_l) and brand_l not in name_l)
    out['has_suspicious_word'] = float(any(w in name_l for w in SUSPICIOUS_WORDS))

    # 7. Возраст товара/продавца — на training: pd.Series default=999 если колонки нет,
    # но если NaN внутри — fillna(-1). Используем NaN-propagating вариант.
    item_age = _fnum('item_time_alive')
    seller_age = _fnum('seller_time_alive')
    # (NaN < 30) → False → 0 (match training: pandas)
    out['new_item_new_seller'] = (
        float(item_age < 30 and seller_age < 180)
        if not (np.isnan(item_age) or np.isnan(seller_age))
        else 0.0
    )
    out['item_seller_age_ratio'] = item_age / (seller_age + 1)  # NaN-propagating

    # 8. Ценовые аномалии: NaN price → log_price=NaN, price_zero=0
    price = _fnum('PriceDiscounted')
    out['price_log'] = float(np.log1p(price))
    out['price_zero'] = float(price == 0) if not np.isnan(price) else 0.0

    return out


def _mode3_loo_lookup(loo_tables: dict, key: str, value: str) -> float:
    """Lookup LOO target encoding с fallback на global_mean (для cold-start
    sellers / brands / categories / names).
    """
    table = loo_tables.get(key, {})
    val = table.get(str(value).strip()) if value not in (None, '') else None
    return float(val) if val is not None else float(loo_tables['global_mean'])


def predict_mode3_alone(
    amm: dict,
    raw_clip_emb: np.ndarray,
    text_svd: np.ndarray,
    tab_features: dict,
    preprocessors,
    name: str = "",
    description: str = "",
    brand: str = "",
    mode3_feature_order: list = None,
    mode3_loo_tables: dict = None,
    mode3_clip_pca: object = None,
) -> float:
    """Извлекает p3 (Mode 3 Deep Deliberation) из AMM pickle с fold averaging.

    HCDM использует Mode 3 standalone как канал социальных сетей; meta-LR не
    применяется. См. § 5.4.6.3 ВКР: Mode 3 (PR-AUC = 0,8052) — сильнейшая
    одиночная модель команды, использованная как канал-якорь композиции
    с весом 0,875.

    Контракт `amm`:
        amm["mode3_folds"] — list[CatBoostClassifier] (5 fold-моделей);
        либо amm["mode3_catboost"] — single CatBoostClassifier (backward-compat).

    Восстановленный feature pipeline (см. `_build_mode3_lookups.py` в репозитории):
        - `mode3_feature_order` — list[str] длины 161 (artifacts/cdsm_v3/mode3_feature_order.json)
        - `mode3_loo_tables` — dict с LOO mappings + global_mean (artifacts/cdsm_v3/mode3_loo_lookups.json)
        - `mode3_clip_pca` — sklearn PCA(n_components=32), отдельный от CDSM v3 PCA
          (artifacts/cdsm_v3/mode3_clip_pca32.joblib)

    Без этих артефактов модель видит нулевой вектор (legacy режим). С ними —
    воспроизводит training-time feature engineering из `safe_ozon_v4_mmd__1_.ipynb`.
    """
    # Backward-compat: если артефакты не переданы, продолжаем legacy-режим
    # (модель видит нулевой вектор → returns base rate).
    if mode3_feature_order is None or mode3_loo_tables is None or mode3_clip_pca is None:
        clip_pca_legacy = preprocessors.clip_pca.transform(raw_clip_emb.reshape(1, -1)).ravel()
        def build_vec_legacy(model) -> np.ndarray:
            feature_names = model.feature_names_
            vec = np.zeros(len(feature_names), dtype=np.float64)
            for i, fn in enumerate(feature_names):
                if fn.startswith("clip_pca_"):
                    j = int(fn.split("_")[-1])
                    vec[i] = clip_pca_legacy[j]
                elif fn.startswith("svd_"):
                    j = int(fn[4:])
                    vec[i] = text_svd[j]
                elif fn in tab_features:
                    val = tab_features[fn]
                    vec[i] = float(val) if val not in (None, "") else 0.0
            return vec.reshape(1, -1)
        m3 = amm.get("mode3_folds") if "mode3_folds" in amm else amm["mode3_catboost"]
        if isinstance(m3, list):
            return float(np.mean([m.predict_proba(build_vec_legacy(m))[0, 1] for m in m3]))
        return float(m3.predict_proba(build_vec_legacy(m3))[0, 1])

    # Full feature engineering pipeline (cell 6-18 of training notebook)
    engineered = _mode3_engineer_features(name, description, brand, tab_features)

    # LOO target encodings (cell 10)
    loo_features = {
        'seller_problem_rate': _mode3_loo_lookup(
            mode3_loo_tables, 'seller_problem_rate', tab_features.get('SellerID', '')
        ),
        'brand_problem_rate': _mode3_loo_lookup(
            mode3_loo_tables, 'brand_problem_rate', brand
        ),
        'category_problem_rate': _mode3_loo_lookup(
            mode3_loo_tables, 'category_problem_rate',
            tab_features.get('CommercialTypeName4', '')
        ),
        'name_problem_rate': _mode3_loo_lookup(
            mode3_loo_tables, 'name_problem_rate', name
        ),
    }

    # CLIP PCA-32 (cell 16 — отдельный PCA, не cdsm_v3/clip_pca.pkl!)
    clip_pca32 = mode3_clip_pca.transform(raw_clip_emb.reshape(1, -1)).ravel()
    has_clip = float(np.linalg.norm(raw_clip_emb) > 1e-6)

    # Assemble vector в правильном порядке (mode3_feature_order)
    # Mode 3 training: train_df[cols_mode3].fillna(-1).values
    vec = np.full(len(mode3_feature_order), -1.0, dtype=np.float64)
    for i, fn in enumerate(mode3_feature_order):
        if fn.startswith('clip_'):
            j = int(fn.split('_')[-1])
            vec[i] = float(clip_pca32[j])
        elif fn.startswith('svd_'):
            j = int(fn[4:])
            vec[i] = float(text_svd[j])
        elif fn == 'has_clip':
            vec[i] = has_clip
        elif fn in engineered:
            v = engineered[fn]
            # NaN из engineered (отсутствующий вход) → -1 как на training
            # после `train_df[cols_mode3].fillna(-1).values`.
            vec[i] = -1.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        elif fn in loo_features:
            vec[i] = loo_features[fn]
        elif fn in tab_features and tab_features[fn] not in (None, ''):
            vec[i] = float(tab_features[fn])
        # else: остаётся -1 (training fillna(-1))

    def fold_avg_predict(folds_or_single) -> float:
        X = vec.reshape(1, -1)
        if isinstance(folds_or_single, list):
            return float(np.mean([m.predict_proba(X)[0, 1] for m in folds_or_single]))
        return float(folds_or_single.predict_proba(X)[0, 1])

    m3 = amm.get("mode3_folds") if "mode3_folds" in amm else amm["mode3_catboost"]
    return fold_avg_predict(m3)


def predict_karina_image(karina_artifact: Dict[str, Any], raw_clip_emb: np.ndarray) -> float:
    """LogReg-инференс на CLIP-эмбеддинге (Baseline C, § 4.1 ВКР).

    Контракт `karina_artifact` (joblib pickle):
        {"scaler": StandardScaler(fit на 512-dim CLIP train+val),
         "lr": LogisticRegression(C=0,5, balanced),
         "feature_dim": 512}
    """
    scaler = karina_artifact["scaler"]
    lr = karina_artifact["lr"]
    X = scaler.transform(raw_clip_emb.reshape(1, -1)).astype("float32")
    return float(lr.predict_proba(X)[0, 1])


def compute_typosquat_features(brand: str, name: str) -> Dict[str, float]:
    """Три Deng-typosquatting признака [Deng et al., 2020] (Group D, § 4.1).

    Используются как расширение tab_inputs для канала p_realestate
    (RMM-typosquat, см. § 5.4.6.9: cross-domain feature-level cooperation).
    """
    try:
        from rapidfuzz.fuzz import partial_ratio
    except ImportError:
        return {"brand_exact": 0.0, "brand_fuzzy": 0.0, "typosquat": 0.0}

    brand_l = (brand or "").strip().lower()
    name_l = (name or "").strip().lower()
    brand_exact = float(bool(brand_l and brand_l in name_l and len(brand_l) > 2))
    brand_fuzzy = float(partial_ratio(brand_l, name_l) / 100.0) if (brand_l and name_l) else 0.0
    typosquat = max(0.0, brand_fuzzy - 0.5 * brand_exact)
    return {
        "brand_exact": brand_exact,
        "brand_fuzzy": brand_fuzzy,
        "typosquat": typosquat,
    }


def compose_hcdm(
    p_social: float,
    p_realestate: float,
    p_fintech: float,
    p_mobile_image: float,
    weights: Dict[str, float] = None,
) -> float:
    """Convex blend четырёх каналов HCDM (§ 5.4.6.3 ВКР).

    Default веса соответствуют итоговой Wolpert-валидированной конфигурации:
        social = 0,875 (канал-якорь, Mode 3)
        mobile_image = 0,075 (ortho diversifier)
        realestate = 0,025 (M2-FE+ с Group D Deng integration)
        fintech = 0,025 (FT-MFF Fusion)
    """
    if weights is None:
        weights = {"social": 0.875, "mobile_image": 0.075, "realestate": 0.025, "fintech": 0.025}
    return (
        weights["social"] * p_social
        + weights["mobile_image"] * p_mobile_image
        + weights["realestate"] * p_realestate
        + weights["fintech"] * p_fintech
    )
