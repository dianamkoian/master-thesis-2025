"""
Production-loaders для Cross-Domain Synthesis Model v3 4-channel ensemble.

Модуль реализует загрузку артефактов и per-channel inference-функции для
CDSMV3Predictor (см. `app/predictor_cdsm.py`). Архитектурно CDSM v3 объединяет
четыре вероятностных канала:

    1) p_cdsm    — synthesis-канал на feature-mix из 8 групп признаков
                    (3-mode CatBoost + 9-мерный meta-vector + LR + sample weights).
                    Полная реализация в `predict_cdsm()` ниже.
    2) p_rmm     — Дианина M2-FE+ (RMM-CatBoost, 636-dim).
                    Полная реализация в `predict_rmm()` ниже.
    3) p_ftmff   — Сонин Fusion (FT-MFF, 92-dim: 42 tab + 25 e5-PCA + 25 CLIP-PCA).
                    Требует Сониного feature-extraction-кода, см. контракт в `predict_ftmff()`.
    4) p_amm     — Альбинин MMD-Thinker v4 (AMM-CatBoost, 3-mode + meta-LR).
                    Требует Альбининого feature-extraction-кода, см. контракт в `predict_amm()`.

Все артефакты ожидаются в `artifacts/cdsm_v3/` со структурой, описанной в
`predictor_cdsm.py` (см. модуль-документацию) и `MODELS.md` § «CDSM v3
production-доукомплектование».

Стратегия инференса:
    • Один CLIP forward pass на запрос (cached внутри CDSMV3Predictor).
    • Один e5-encoding на запрос (cached).
    • Каждая под-модель получает уже-извлечённые dense эмбеддинги
      и применяет свой post-processing (PCA-25 / raw-512 / structural).

В graceful-fallback-режиме (артефактов нет) CDSMV3Predictor использует
`D2VCatBoostPredictor` baseline, см. `predictor_cdsm.py:_activate_fallback()`.
"""
from __future__ import annotations

import io
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Универсальные loader-функции
# ─────────────────────────────────────────────────────────────────────────────

def load_pkl(path: Path) -> Any:
    """Загрузка pickle/joblib-артефакта (scaler, PCA, vectorizer, meta-LR)."""
    import joblib
    return joblib.load(path)


def load_catboost(path: Path):
    """Загрузка обученной CatBoost-модели из .cbm-файла."""
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def load_lr_with_scaler(coefs_path: Path) -> "MetaLR":
    """Загрузка финального L1-Logistic-Regression-блендинга (coefs + scaler-статистики)
    из `final_lr_coefs.json`.

    Ожидаемая структура JSON:
        {
            "coefs": {"cdsm": float, "rmm": float, "ftmff": float, "amm": float},
            "intercept": float,
            "scaler": {"mean": [4 floats], "std": [4 floats]},
            "best_C": float
        }
    """
    with coefs_path.open() as f:
        cfg = json.load(f)
    return MetaLR(
        coefs=cfg["coefs"],
        intercept=float(cfg.get("intercept", 0.0)),
        scaler_mean=np.asarray(cfg["scaler"]["mean"], dtype=np.float64),
        scaler_std=np.asarray(cfg["scaler"]["std"], dtype=np.float64),
    )


@dataclass
class MetaLR:
    """Финальный L1-Logistic-Regression-блендинг каналов CDSM v3.

    Поддерживает 3 канала (cdsm + rmm + ftmff, без AMM — production-режим при
    отсутствии Альбининой модели) или 4 канала (полный CDSM v3 4-channel, если
    `amm_thinker.pkl` доступен). Решение принимается по наличию ключа `amm`
    в `coefs` dict и по совпадению длины `scaler_mean` с числом каналов.
    """
    coefs: dict[str, float]
    intercept: float
    scaler_mean: np.ndarray
    scaler_std: np.ndarray

    def __call__(self, p_cdsm: float, p_rmm: float, p_ftmff: float,
                  p_amm: Optional[float] = None) -> float:
        if "amm" in self.coefs and p_amm is not None:
            x = np.array([p_cdsm, p_rmm, p_ftmff, p_amm], dtype=np.float64)
            channel_keys = ["cdsm", "rmm", "ftmff", "amm"]
        else:
            x = np.array([p_cdsm, p_rmm, p_ftmff], dtype=np.float64)
            channel_keys = ["cdsm", "rmm", "ftmff"]
        z = (x - self.scaler_mean) / (self.scaler_std + 1e-12)
        logit = float(self.intercept) + sum(self.coefs[k] * z[i] for i, k in enumerate(channel_keys))
        return float(1.0 / (1.0 + np.exp(-logit)))


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка под-моделей (4 под-классификатора + meta-LR + общие preprocessors)
# ─────────────────────────────────────────────────────────────────────────────

def load_amm_thinker(path: Path):
    """Загрузка Альбининой MMD-Thinker v4 (multi-mode ensemble + meta-LR).

    Внутренняя структура pickle-объекта (контракт Альбины):
        {
            "mode1_catboost": CatBoostClassifier,  # Mode 1 — tab only
            "mode2_catboost": CatBoostClassifier,  # Mode 2 — tab + TF-IDF SVD
            "mode3_catboost": CatBoostClassifier,  # Mode 3 — tab + CLIP-PCA
            "meta_lr":        sklearn LogisticRegression на 9-мерном meta-vector,
            "feature_cols": {
                "mode1": [str, ...],
                "mode2": [str, ...],
                "mode3": [str, ...],
            },
            "sample_weight_uncertainty_threshold": float,  # формула A_sample из Wu & Fu 2025
        }
    """
    with path.open("rb") as f:
        return pickle.load(f)


def load_cdsm_pipeline(cdsm_dir: Path):
    """Загрузка внутреннего CDSM-канала: 5 fold-моделей per mode + meta-LR.

    Production-deploy использует **5 fold-моделей** для каждого из 3 mode
    (итого 15 .cbm) и усредняет их predictions — это даёт **идеальное**
    воспроизведение OOF-style probas, на которых обучалась meta-LR в § 5.4.6.6 ВКР.
    Без такого усреднения возникает distribution mismatch между training
    (OOF-cv-averaged probas) и production-inference (single-fit probas), что
    приводит к небольшим, но устойчивым отклонениям headline-метрик.

    Структура каталога:
        cdsm_dir/
            ├── cdsm_mode1_fold{0,1,2,3,4}.cbm  ← 5 fold-моделей Mode 1
            ├── cdsm_mode2_fold{0,1,2,3,4}.cbm  ← 5 fold-моделей Mode 2
            ├── cdsm_mode3_fold{0,1,2,3,4}.cbm  ← 5 fold-моделей Mode 3
            └── cdsm_meta_lr.pkl                ← meta-LR на 9-мерном meta-vector
    """
    def load_fold_models(mode_name: str) -> list:
        models = []
        for fi in range(5):
            path = cdsm_dir / f"cdsm_{mode_name}_fold{fi}.cbm"
            if not path.is_file():
                raise FileNotFoundError(f"Missing fold model: {path}")
            models.append(load_catboost(path))
        return models

    return CDSMPipeline(
        mode1_folds=load_fold_models("mode1"),
        mode2_folds=load_fold_models("mode2"),
        mode3_folds=load_fold_models("mode3"),
        meta_lr=load_pkl(cdsm_dir / "cdsm_meta_lr.pkl"),
    )


@dataclass
class CDSMPipeline:
    """Внутренний CDSM-канал: 5 fold-моделей per mode + meta-LR + sample weighting.

    Архитектура — three-mode adaptive deep processing (Wu & Fu, MMD-Thinker, 2025;
    см. § 5.4.6.1 ВКР). Mode 1 — Quick (44-dim tabular + identity-deception);
    Mode 2 — Semantic (Mode 1 + TF-IDF SVD-50); Mode 3 — Deep (Mode 2 + CLIP-PCA-25
    + CLIP-derived structural + cross-modal KL).

    Meta-LR работает на 9-мерном расширенном представлении:
        [p1, p2, p3, |p1-p2|, |p1-p3|, |p2-p3|, p1·p2, p1·p3, p2·p3]
    + sample-difficulty weighting w_i = 1 + (1 − 2|p1 − 0,5|).

    Production-inference: каждое из p_mode{1,2,3} вычисляется как **среднее**
    по 5 fold-моделям. Это reproduces OOF-style probas, на которых обучалась
    meta-LR, обеспечивая идеальное соответствие headline-метрикам ВКР § 5.4.6.6.
    """
    mode1_folds: list  # list[CatBoostClassifier], 5 fold-моделей Mode 1
    mode2_folds: list  # list[CatBoostClassifier], 5 fold-моделей Mode 2
    mode3_folds: list  # list[CatBoostClassifier], 5 fold-моделей Mode 3
    meta_lr: Any       # sklearn LogisticRegression на 9-мерном meta-vector

    @staticmethod
    def _avg_fold_predict(models: list, features: np.ndarray) -> float:
        """Усреднение predict_proba по списку fold-моделей (OOF-style)."""
        X = features.reshape(1, -1)
        return float(np.mean([m.predict_proba(X)[0, 1] for m in models]))

    def predict_proba_one(
        self,
        features_mode1: np.ndarray,  # shape (44,)
        features_mode2: np.ndarray,  # shape (94,) = 44 + 50 SVD
        features_mode3: np.ndarray,  # shape (149,) = 94 + 25 CLIP-PCA + 4 structural + 25 e5-PCA + 1 KL
    ) -> float:
        # Прогон через 5-fold-average для каждого Mode
        p1 = self._avg_fold_predict(self.mode1_folds, features_mode1)
        p2 = self._avg_fold_predict(self.mode2_folds, features_mode2)
        p3 = self._avg_fold_predict(self.mode3_folds, features_mode3)
        # 9-мерное meta-представление в canonical-порядке
        # `meta_features` в notebook 08 cell 20: [p1, p2, p3, |p1-p2|, |p2-p3|, |p1-p3|, p1*p2, p1*p3, p2*p3]
        meta = np.array([
            p1, p2, p3,
            abs(p1 - p2), abs(p2 - p3), abs(p1 - p3),
            p1 * p2, p1 * p3, p2 * p3,
        ], dtype=np.float64).reshape(1, -1)
        # `meta_lr` — sklearn Pipeline(scaler → lr), сохранённый build_cdsm_meta_lr.py;
        # Pipeline.predict_proba автоматически применяет StandardScaler перед LR.
        return float(self.meta_lr.predict_proba(meta)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel predict-функции
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SharedPreprocessors:
    """Общие препроцессоры, разделяемые между каналами CDSM v3.

    CLIP / multilingual-e5-small / TF-IDF — тяжёлые модели; их forward pass
    выполняется один раз на запрос, результат переиспользуется во всех под-каналах.
    """
    tfidf_vectorizer: Any   # sklearn TF-IDF + TruncatedSVD-50 (общекомандный)
    e5_pca: Any             # PCA-25 поверх multilingual-e5-small (Сонин)
    clip_pca: Any           # PCA-25 поверх CLIP ViT-B/32 (Сонин-Дианин)
    img_scaler: Any         # StandardScaler для raw-512 CLIP (Дианина M2-FE+)


def extract_text_svd(text: str, preprocessors: SharedPreprocessors) -> np.ndarray:
    """Извлечение TF-IDF + TruncatedSVD-50 представления текста.

    Совместимо с двумя вариантами artefact-а:
    - sklearn Pipeline(tfidf, svd) — старая команная упаковка → .named_steps["svd"]
    - Просто TfidfVectorizer без SVD-шага → возвращаем zeros(50) как graceful degradation
      (downstream CatBoost обработает нулевые SVD-features через OTS).
    """
    sparse = preprocessors.tfidf_vectorizer.transform([text])
    if hasattr(preprocessors.tfidf_vectorizer, "named_steps") and "svd" in preprocessors.tfidf_vectorizer.named_steps:
        return preprocessors.tfidf_vectorizer.named_steps["svd"].transform(sparse).ravel().astype(np.float64)
    # Fallback: возвращаем zeros(50) — features svd_0..svd_49 будут нулевыми,
    # CatBoost-модели Mode 2/3 это обработают через OTS как missing.
    return np.zeros(50, dtype=np.float64)


def extract_clip_features(image_bytes: bytes, clip_model, clip_processor) -> np.ndarray:
    """Один CLIP forward-pass: bytes → raw 512-dim эмбеддинг.

    Этот метод ожидается, что будет вызван CDSMV3Predictor **один раз на запрос**,
    результат кэшируется и передаётся во все four under-channel-predict-функции.
    """
    from PIL import Image
    import torch
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    # Совместимость с разными версиями transformers:
    # в старых версиях возвращается tensor, в новых — ModelOutput с .pooler_output
    if hasattr(emb, "pooler_output"):
        emb = emb.pooler_output
    elif hasattr(emb, "last_hidden_state"):
        emb = emb.last_hidden_state[:, 0, :]
    return emb.cpu().numpy().ravel().astype(np.float64)


def extract_e5_embedding(text: str, e5_model, e5_tokenizer) -> np.ndarray:
    """Один e5-encoding: text → 384-dim dense эмбеддинг (multilingual-e5-small).

    Кэшируется на уровне `CDSMV3Predictor.predict()` так же, как CLIP-эмбеддинг.
    """
    import torch
    prefix = "passage: "
    enc = e5_tokenizer([prefix + text], padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        out = e5_model(**enc)
    # mean pooling + L2-нормализация (стандартный e5-протокол)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return pooled.cpu().numpy().ravel().astype(np.float64)


# ─── RMM-канал (Дианина M2-FE+, см. § 4.4.3.2 ВКР) ─────────────────────────

def predict_rmm(
    model,                              # CatBoostClassifier RMM
    raw_clip_emb: np.ndarray,           # shape (512,) — raw CLIP, без PCA
    text_svd: np.ndarray,               # shape (50,) — TF-IDF SVD-50
    tab_features: dict,                 # все Дианиные FE-фичи + 38 командных
    preprocessors: SharedPreprocessors,
) -> float:
    """Production-инференс Дианиной M2-FE+ на едином признаковом пространстве 636-dim:
        51 табличный + 512 CLIP raw + 50 TF-IDF SVD + 23 FE-фичи
            (text-stats, brand-агрегаты, coherence, interactions,
             CLIP-derived structural, cat-interactions).

    Структура feature-вектора задана `model.feature_names_` — порядок жёстко
    зафиксирован при обучении в `notebooks/02_reproduction_fixed.ipynb`.
    """
    scaled_clip = preprocessors.img_scaler.transform(raw_clip_emb.reshape(1, -1)).ravel()
    feature_names = model.feature_names_
    cat_indices = list(model.get_cat_feature_indices())
    cat_set = set(cat_indices)
    row: list = []
    for i, fn in enumerate(feature_names):
        if fn.startswith("img_"):
            j = int(fn[4:])
            row.append(float(scaled_clip[j]))
        elif fn.startswith("svd_"):
            j = int(fn[4:])
            row.append(float(text_svd[j]))
        elif i in cat_set:
            val = tab_features.get(fn, "nan")
            row.append(str(val) if val not in (None, "") else "nan")
        elif fn in tab_features:
            val = tab_features[fn]
            row.append(float(val) if val not in (None, "") else 0.0)
        else:
            row.append(0.0)
    # Используем numpy object-array — сохраняет mixed types (str для cat, float для numeric).
    import pandas as _pd
    arr = np.array([row], dtype=object)
    X = _pd.DataFrame(arr, columns=list(feature_names))
    for i, fn in enumerate(feature_names):
        if i in cat_set:
            X[fn] = X[fn].astype(str)
        else:
            X[fn] = _pd.to_numeric(X[fn], errors="coerce").fillna(0.0).astype("float64")
    return float(model.predict_proba(X)[0, 1])


# ─── FT-MFF-канал (Сонин Fusion, см. § 4.2 / § 5.4.5.2 ВКР) ────────────────

def predict_ftmff(
    model,                              # CatBoostClassifier FT-MFF
    raw_clip_emb: np.ndarray,           # shape (512,)
    raw_e5_emb: np.ndarray,             # shape (384,) — multilingual-e5-small
    tab_features: dict,                 # 42 финтех-tabular признака
    preprocessors: SharedPreprocessors,
) -> float:
    """Production-инференс Сониного Fusion CatBoost на 92-dim признаковом пространстве:
        42 финтех-tabular + 25 CLIP-PCA-25 + 25 e5-PCA-25.

    Контракт Сони (см. `Глава_5_финал/github_sync/multimodal_team_summary.json`
    → `Fusion_team`): CatBoost iterations=1000, depth=6, scale_pos_weight≈15,40.
    Reference: `Глава_5_финал/github_sync/07_stacking_pilot_4domains.py`.

    ⚠️ Точный порядок признаков и имена tabular-колонок должны соответствовать
    feature_names_ обученной Сониной модели. При несовпадении CatBoost OTS
    выбросит RuntimeError. Список 42 финтех-tabular-колонок зафиксирован
    в `team_split_meta.json` → `team_feature_cols` (38 общекомандных) плюс
    четыре финтех-специфичных (`brand_name`, `description`, `name_rus`,
    `CommercialTypeName4`) — последние подаются Соней через CatBoost-OTS как
    cat-features. См. её собственную Главу 4 для полного списка.
    """
    clip_pca = preprocessors.clip_pca.transform(raw_clip_emb.reshape(1, -1)).ravel()
    e5_pca = preprocessors.e5_pca.transform(raw_e5_emb.reshape(1, -1)).ravel()
    feature_names = model.feature_names_
    cat_indices = list(model.get_cat_feature_indices())
    cat_set = set(cat_indices)
    row: list = []
    for i, fn in enumerate(feature_names):
        if fn.startswith("clip_pca_"):
            j = int(fn.split("_")[-1])
            row.append(float(clip_pca[j]))
        elif fn.startswith("t_pca_") or fn.startswith("e5_pca_"):
            j = int(fn.split("_")[-1])
            row.append(float(e5_pca[j]))
        elif i in cat_set:
            val = tab_features.get(fn, "nan")
            row.append(str(val) if val not in (None, "") else "nan")
        elif fn in tab_features:
            val = tab_features[fn]
            row.append(float(val) if val not in (None, "") else 0.0)
        else:
            row.append(0.0)
    # Используем numpy object-array — сохраняет mixed types (str для cat, float для numeric).
    import pandas as _pd
    arr = np.array([row], dtype=object)
    X = _pd.DataFrame(arr, columns=list(feature_names))
    for i, fn in enumerate(feature_names):
        if i in cat_set:
            X[fn] = X[fn].astype(str)
        else:
            X[fn] = _pd.to_numeric(X[fn], errors="coerce").fillna(0.0).astype("float64")
    return float(model.predict_proba(X)[0, 1])


# ─── AMM-канал (Альбинин MMD-Thinker v4, см. § 4.3 / § 5.4.5.2 ВКР) ────────

def predict_amm(
    amm,                                # pickle-объект из `load_amm_thinker()`
    raw_clip_emb: np.ndarray,
    text_svd: np.ndarray,
    tab_features: dict,
    preprocessors: SharedPreprocessors,
) -> float:
    """Production-инференс Альбининой MMD-Thinker v4: three-mode CatBoost +
    meta-LR на 9-мерном расширенном представлении.

    Архитектура (см. § 5.4.6.1 ВКР и Wu & Fu 2025):
        Mode 1 (Quick):       только табличные финтех-признаки
        Mode 2 (Semantic):    Mode 1 + 50 TF-IDF SVD
        Mode 3 (Deep):        Mode 2 + 25 CLIP-PCA-25
        Meta-LR:              [p1, p2, p3, |p_i-p_j| × 3, p_i·p_j × 3]
                              + adaptive sample weighting через uncertainty Mode 1.

    Контракт `amm` (pickle-объект):
        amm["mode1_catboost"], amm["mode2_catboost"], amm["mode3_catboost"]
            — обученные CatBoostClassifier
        amm["meta_lr"]       — sklearn LogisticRegression (C=0.5, class_weight=balanced)
        amm["feature_cols"]  — dict со списком feature_names_ для каждого mode

    ⚠️ Точная feature-extraction-логика (включая Альбинину обработку категориальных
    признаков и формулу A_sample для sample weighting) ожидается в
    `amm["feature_cols"]` и применяется через стандартный `CatBoostClassifier.predict_proba()`.
    Полный pickle-контракт зафиксирован в `MODELS.md` § «AMM-Thinker v4
    production-доукомплектование (А. Бахтиарова)».
    """
    clip_pca = preprocessors.clip_pca.transform(raw_clip_emb.reshape(1, -1)).ravel()

    def build_vec(model) -> np.ndarray:
        feature_names = model.feature_names_
        vec = np.zeros(len(feature_names), dtype=np.float64)
        for i, fn in enumerate(feature_names):
            if fn.startswith("clip_pca_"):
                j = int(fn.split("_")[-1])
                vec[i] = clip_pca[j]
            elif fn.startswith("svd_"):
                j = int(fn[4:])
                vec[i] = text_svd[j]
            elif fn in tab_features:
                val = tab_features[fn]
                vec[i] = float(val) if val not in (None, "") else 0.0
        return vec.reshape(1, -1)

    def fold_avg_predict(folds_or_single) -> float:
        """5-fold averaging если folds — list, иначе single predict."""
        if isinstance(folds_or_single, list):
            return float(np.mean([m.predict_proba(build_vec(m))[0, 1] for m in folds_or_single]))
        return float(folds_or_single.predict_proba(build_vec(folds_or_single))[0, 1])

    # Используем 5-fold averaging если доступен mode{1,2,3}_folds (canonical Albina patched ноут),
    # иначе fallback на single mode{1,2,3}_catboost (backward-compat с моим reproduction).
    m1 = amm.get("mode1_folds") if "mode1_folds" in amm else amm["mode1_catboost"]
    m2 = amm.get("mode2_folds") if "mode2_folds" in amm else amm["mode2_catboost"]
    m3 = amm.get("mode3_folds") if "mode3_folds" in amm else amm["mode3_catboost"]
    p1 = fold_avg_predict(m1)
    p2 = fold_avg_predict(m2)
    p3 = fold_avg_predict(m3)
    meta = np.array([
        p1, p2, p3,
        abs(p1 - p2), abs(p2 - p3), abs(p1 - p3),
        p1 * p2, p1 * p3, p2 * p3,
    ], dtype=np.float64).reshape(1, -1)
    return float(amm["meta_lr"].predict_proba(meta)[0, 1])


# ─── CDSM synthesis-канал (Дианин 3-mode CatBoost + meta-LR, см. § 5.4.6) ──

def predict_cdsm(
    cdsm_pipeline: CDSMPipeline,
    raw_clip_emb: np.ndarray,
    raw_e5_emb: np.ndarray,
    text_svd: np.ndarray,
    tab_features: dict,
    cross_modal_kl: float,              # OOF KL(text‖tab) feature (Group I, § 5.4.6.2)
    structural_features: dict,           # 4 CLIP-derived structural (Group H) + brand-агрегаты + typosquat
    preprocessors: SharedPreprocessors,
) -> float:
    """Production-инференс собственного CDSM-канала (3-mode + meta-LR).

    Признаковое пространство Mode i:
        Mode 1: A (38 tab) + C (3 FADAML) + D (3 typosquat-Deng) = 44 признака
        Mode 2: Mode 1 + E (50 TF-IDF SVD) = 94 признака
        Mode 3: Mode 2 + F (25 e5-PCA) + G (25 CLIP-PCA) + H (4 CLIP-structural)
                       + I (1 cross-modal KL) = 149 признаков

    См. § 5.4.6.2 ВКР (Таблица 5.4.6.а — провенанс групп признаков) для полного
    описания каждой feature-группы и её атрибуции к исходному домену.
    """
    clip_pca = preprocessors.clip_pca.transform(raw_clip_emb.reshape(1, -1)).ravel()
    e5_pca = preprocessors.e5_pca.transform(raw_e5_emb.reshape(1, -1)).ravel()

    def build_mode_vec(model, includes_text: bool, includes_image: bool) -> np.ndarray:
        # Все fold-модели имеют одинаковый feature_names_ (один и тот же
        # MODE_GROUPS-набор колонок), берём первой fold-модели для построения вектора.
        feature_names = model.feature_names_
        vec = np.zeros(len(feature_names), dtype=np.float64)
        for i, fn in enumerate(feature_names):
            if includes_image and fn.startswith("clip_pca_"):
                vec[i] = clip_pca[int(fn.split("_")[-1])]
            elif includes_image and fn.startswith("clip_struct_"):
                vec[i] = structural_features.get(fn, 0.0)
            elif includes_image and fn == "cross_modal_kl":
                vec[i] = float(cross_modal_kl)
            elif includes_text and fn.startswith("svd_"):
                vec[i] = text_svd[int(fn[4:])]
            elif includes_text and fn.startswith("e5_pca_"):
                vec[i] = e5_pca[int(fn.split("_")[-1])]
            elif fn in tab_features:
                vec[i] = float(tab_features[fn] or 0.0)
            elif fn in structural_features:
                vec[i] = float(structural_features[fn] or 0.0)
            # missing → 0 (CatBoost обрабатывает через ordered target statistics)
        return vec

    return cdsm_pipeline.predict_proba_one(
        features_mode1=build_mode_vec(cdsm_pipeline.mode1_folds[0], includes_text=False, includes_image=False),
        features_mode2=build_mode_vec(cdsm_pipeline.mode2_folds[0], includes_text=True, includes_image=False),
        features_mode3=build_mode_vec(cdsm_pipeline.mode3_folds[0], includes_text=True, includes_image=True),
    )
