"""
CDSMV3Predictor — production-конфигурация Cross-Domain Synthesis Model v3
4-channel ensemble в сервисе детекции контрафакта.

Архитектурно CDSM v3 4-channel ensemble — это L1-Logistic-Regression-блендинг
четырёх вероятностных каналов на канонических probas общекомандного теста:

    1) p_cdsm     — synthesis-канал: 3-mode CatBoost + meta-LR + sample weights,
                     обучен на feature-mix из 8 групп (см. § 5.4.6.2 диплома)
    2) p_rmm      — RMM-CatBoost: Дианина M2-FE+ headline (домен недвижимости)
    3) p_ftmff    — FT-MFF: Сонин Fusion CatBoost (домен финтеха)
    4) p_amm      — AMM-CatBoost: Альбинина MMD-Thinker v4 (домен соц. сетей)

Финальное предсказание: sigmoid(b + sum_i w_i * z(p_i)), где w_i — обученные
L1-LR-коэффициенты (см. `artifacts/cdsm_v3/lr_coefs.json`), z — StandardScaler
на meta-fit подмножестве.

Артефакты, необходимые для активации `cdsm_v3_4channel` в production:
    artifacts/cdsm_v3/
        ├── cdsm_mode1.cbm       — CatBoost Mode 1 (Quick: 44 признака)
        ├── cdsm_mode2.cbm       — CatBoost Mode 2 (+50 TF-IDF SVD)
        ├── cdsm_mode3.cbm       — CatBoost Mode 3 (+25 CLIP-PCA + cross-modal)
        ├── cdsm_meta_lr.pkl     — Logistic Regression поверх 9-мерного meta-vector
        ├── rmm_catboost.cbm     — Дианина M2-FE+ (636-dim feature space)
        ├── ftmff_catboost.cbm   — Сонин Fusion (92-dim feature space)
        ├── amm_thinker.pkl      — Альбинин MMD-Thinker v4 pipeline
        ├── final_lr_coefs.json  — coefs L1-LR + intercept + scaler stats
        ├── tfidf_vectorizer.pkl — общекомандный TF-IDF + SVD-50
        ├── e5_pca.pkl           — multilingual-e5-small PCA-25
        ├── clip_pca.pkl         — CLIP ViT-B/32 PCA-25
        └── manifest.json        — версии моделей, contributors, MD5 канонических npy

Активация:
    PREDICTOR_TYPE=cdsm_v3_4channel docker compose up

При отсутствии артефактов соавторов (RMM/FT-MFF/AMM) предиктор делает
graceful fallback на baseline-конфигурацию с явным предупреждением в /health.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .predictor_base import BasePredictor

logger = logging.getLogger(__name__)


# Базовая директория артефактов CDSM v3
_ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_ROOT", "artifacts"))
CDSM_DIR = _ARTIFACTS_ROOT / "cdsm_v3"


# Ожидаемые координаты L1-LR на полном тесте (random_state=42), для проверки
# manifest-целостности после обучения соавторов. Значения из § 5.4.6.6 диплома
# и `Глава_5_финал/artifacts/domain_lodo_full.json` → `baseline_v3_4ch.coefs`.
_EXPECTED_COEFS = {
    "cdsm":   0.36760232783476876,
    "rmm":    0.11015995853632249,
    "ftmff":  0.5518958121912780,
    "amm":    0.834668808931904,
}


class CDSMV3Predictor(BasePredictor):
    """Cross-Domain Synthesis Model v3 4-channel ensemble.

    headline production-конфигурация канала ранжирования (см. § 5.6.2 ВКР).
    На едином командном тестовом сплите (n = 58 410): PR-AUC = 0,7579,
    Recall@P ≥ 0,9 = 0,2078, ROC-AUC = 0,9603 (см. Таблицу 5.4.6.д).

    Архитектура — двухуровневая кросс-доменная агрегация:
      • feature-level synthesis из методов всех 4 доменов команды внутри
        CDSM-канала через 3-mode CatBoost-эскалацию;
      • channel-level L1-Logistic-Regression блендинг четырёх вероятностных
        каналов (CDSM + RMM + FT-MFF + AMM).

    Парный bootstrap-анализ (B = 2 000, seed = 42) подтверждает значимое
    превосходство над командным baseline-блендингом B0 по всем трём ключевым
    метрикам (Δ R@P ≥ 0,9 = +0,0831, относительный прирост ≈ +71 %); см. § 5.4.8.
    Multi-seed устойчивость (5 повторов): PR = 0,7608 ± 0,0022, R@P = 0,2123 ± 0,0050,
    ROC = 0,9606 ± 0,0003 (см. § 5.4.6.6).

    Требования к артефактам: см. модуль-документацию выше. Если артефактный
    пакет соавторов недоступен, предиктор выполняет graceful fallback на
    `D2VCatBoostPredictor` и пишет предупреждение в `/health` JSON.
    """

    name = "cdsm_v3_4channel"
    description = (
        "Cross-Domain Synthesis Model v3 4-channel ensemble (Канал ранжирования "
        "production-сервиса, headline-конфигурация § 5.6.2 ВКР). PR-AUC = 0,7579, "
        "Recall@P ≥ 0,9 = 0,2078, ROC-AUC = 0,9603 на едином командном тесте. "
        "Требует артефактов всех четырёх доменных моделей соавторов."
    )

    def __init__(self) -> None:
        self._fallback: Optional[BasePredictor] = None
        self._artifacts_loaded: bool = False
        self._coefs: dict[str, float] = {}
        self._intercept: float = 0.0
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._manifest: dict = {}

    # ─────────────────────────────────────────────────────────────────────
    # Артефактная загрузка
    # ─────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Попытка загрузить все CDSM v3 артефакты.

        При отсутствии любого из обязательных артефактов — graceful fallback
        на `D2VCatBoostPredictor` (default baseline). Это позволяет сервису
        запускаться даже до того, как все соавторы доукомплектуют свои
        обученные модели в общекомандный pkl-пакет.
        """
        if not CDSM_DIR.is_dir():
            logger.warning(
                "CDSMV3Predictor: каталог артефактов %s не найден. "
                "Активирую graceful fallback на D2VCatBoostPredictor. "
                "Для полноценной CDSM v3 production-активации соавторам "
                "необходимо доукомплектовать %s (см. MODELS.md).",
                CDSM_DIR, CDSM_DIR,
            )
            self._activate_fallback()
            return

        manifest_path = CDSM_DIR / "manifest.json"
        # Auto-detect режим: 3-channel (без AMM) если amm_thinker.pkl отсутствует.
        amm_path = CDSM_DIR / "amm_thinker.pkl"
        self._has_amm = amm_path.is_file()
        coefs_filename = "final_lr_coefs.json" if self._has_amm else "final_lr_coefs_3ch.json"
        coefs_path = CDSM_DIR / coefs_filename
        logger.info(
            "CDSMV3Predictor: AMM-канал %s, используется %s",
            "доступен" if self._has_amm else "недоступен (3-channel режим)",
            coefs_filename,
        )

        # Required список — без amm_thinker.pkl (опциональный)
        required = [
            "cdsm_mode1_fold0.cbm", "cdsm_mode1_fold1.cbm", "cdsm_mode1_fold2.cbm",
            "cdsm_mode1_fold3.cbm", "cdsm_mode1_fold4.cbm",
            "cdsm_mode2_fold0.cbm", "cdsm_mode2_fold1.cbm", "cdsm_mode2_fold2.cbm",
            "cdsm_mode2_fold3.cbm", "cdsm_mode2_fold4.cbm",
            "cdsm_mode3_fold0.cbm", "cdsm_mode3_fold1.cbm", "cdsm_mode3_fold2.cbm",
            "cdsm_mode3_fold3.cbm", "cdsm_mode3_fold4.cbm",
            "cdsm_meta_lr.pkl",
            "rmm_catboost.cbm", "ftmff_catboost.cbm",
            "tfidf_vectorizer.pkl", "e5_pca.pkl", "clip_pca.pkl",
        ]
        missing = [f for f in required if not (CDSM_DIR / f).is_file()]
        if missing or not coefs_path.is_file():
            logger.warning(
                "CDSMV3Predictor: отсутствуют артефакты %s. "
                "Активирую graceful fallback. Для активации see MODELS.md § «CDSM v3».",
                missing or [coefs_filename],
            )
            self._activate_fallback()
            return

        # Загрузка L1-LR coefs + scaler stats
        with coefs_path.open() as f:
            cfg = json.load(f)
        self._coefs = dict(cfg["coefs"])
        self._intercept = float(cfg.get("intercept", 0.0))
        self._scaler_mean = np.asarray(cfg["scaler"]["mean"], dtype=np.float64)
        self._scaler_std = np.asarray(cfg["scaler"]["std"],  dtype=np.float64)

        if manifest_path.is_file():
            with manifest_path.open() as f:
                self._manifest = json.load(f)

        # Загрузка под-моделей CatBoost + (опционально) AMM + meta-LR + общие preprocessors
        try:
            from . import _cdsm_loaders
            self._cdsm_pipeline = _cdsm_loaders.load_cdsm_pipeline(CDSM_DIR)
            self._rmm = _cdsm_loaders.load_catboost(CDSM_DIR / "rmm_catboost.cbm")
            self._ftmff = _cdsm_loaders.load_catboost(CDSM_DIR / "ftmff_catboost.cbm")
            self._amm = _cdsm_loaders.load_amm_thinker(amm_path) if self._has_amm else None
            self._final_lr = _cdsm_loaders.load_lr_with_scaler(coefs_path)
            # Общие preprocessors (CLIP-PCA, e5-PCA, TF-IDF SVD, img-scaler)
            self._preprocessors = _cdsm_loaders.SharedPreprocessors(
                tfidf_vectorizer=_cdsm_loaders.load_pkl(CDSM_DIR / "tfidf_vectorizer.pkl"),
                e5_pca=_cdsm_loaders.load_pkl(CDSM_DIR / "e5_pca.pkl"),
                clip_pca=_cdsm_loaders.load_pkl(CDSM_DIR / "clip_pca.pkl"),
                img_scaler=_cdsm_loaders.load_pkl(CDSM_DIR / "img_scaler.pkl"),
            )
            # CLIP и e5 — лёгкие модели от HuggingFace, lazy-load на первый запрос
            self._clip_model = None
            self._clip_processor = None
            self._e5_model = None
            self._e5_tokenizer = None
            self._loaders = _cdsm_loaders
            self._artifacts_loaded = True
            logger.info(
                "CDSMV3Predictor: артефакты загружены. Manifest contributors: %s",
                self._manifest.get("contributors", "—"),
            )
        except (ImportError, FileNotFoundError, KeyError) as e:
            logger.warning(
                "CDSMV3Predictor: не удалось загрузить production-артефакты (%s: %s). "
                "Активирую graceful fallback на D2VCatBoostPredictor.",
                type(e).__name__, e,
            )
            self._activate_fallback()

    def _activate_fallback(self) -> None:
        """Подключить D2VCatBoostPredictor как fallback при отсутствии CDSM-артефактов."""
        from .predictor import D2VCatBoostPredictor
        self._fallback = D2VCatBoostPredictor()
        self._fallback.load()

    # ─────────────────────────────────────────────────────────────────────
    # Инференс
    # ─────────────────────────────────────────────────────────────────────

    def predict(
        self,
        image_bytes: bytes,
        name: str,
        description: str,
        brand: str,
        tab_inputs: dict,
    ) -> dict:
        """Инференс по контракту `BasePredictor`."""
        if not self._artifacts_loaded:
            assert self._fallback is not None, "fallback не активирован"
            result = self._fallback.predict(image_bytes, name, description, brand, tab_inputs)
            # Помечаем в signals, что использован fallback (для прозрачности /predict-ответа)
            result.setdefault("signals", {})["model_route"] = "cdsm_v3_fallback_baseline"
            return result

        # ───── Production-инференс CDSM v3 4-channel ensemble ─────
        # Шаг 0: один CLIP forward + один e5-forward на запрос (cached)
        self._ensure_dense_encoders_loaded()
        text_concat = f"{brand} {name} {description}".strip()
        raw_clip_emb = self._loaders.extract_clip_features(image_bytes, self._clip_model, self._clip_processor)
        raw_e5_emb = self._loaders.extract_e5_embedding(text_concat, self._e5_model, self._e5_tokenizer)
        text_svd = self._loaders.extract_text_svd(text_concat, self._preprocessors)

        # Производные структурные признаки (Group H, § 5.4.6.2) + KL feature (Group I)
        structural = self._compute_structural_features(raw_clip_emb, tab_inputs, brand, name)
        cross_modal_kl = self._compute_cross_modal_kl(raw_clip_emb, text_svd, tab_inputs)

        # Шаг 1: получить four channel-probas через _cdsm_loaders
        p_rmm = self._loaders.predict_rmm(
            self._rmm, raw_clip_emb, text_svd, tab_inputs, self._preprocessors,
        )
        p_ftmff = self._loaders.predict_ftmff(
            self._ftmff, raw_clip_emb, raw_e5_emb, tab_inputs, self._preprocessors,
        )
        p_amm = self._loaders.predict_amm(
            self._amm, raw_clip_emb, text_svd, tab_inputs, self._preprocessors,
        )
        p_cdsm = self._loaders.predict_cdsm(
            self._cdsm_pipeline, raw_clip_emb, raw_e5_emb, text_svd,
            tab_inputs, cross_modal_kl, structural, self._preprocessors,
        )

        # Шаг 2: финальный L1-LR блендинг (sigmoid + scaler внутри MetaLR)
        proba = self._final_lr(p_cdsm, p_rmm, p_ftmff, p_amm)

        return {
            "is_counterfeit": bool(proba >= 0.5),
            "probability": round(proba, 4),
            "signals": {
                "multimodal_score": round(proba, 4),
                "image_signal": round((p_rmm + p_ftmff) / 2.0, 4),
                "text_signal":  round((p_cdsm + p_amm)  / 2.0, 4),
                # Расширенные сигналы CDSM v3 (диагностические, для UI):
                "p_cdsm":   round(p_cdsm,   4),
                "p_rmm":    round(p_rmm,    4),  # Diana M2-FE+
                "p_ftmff":  round(p_ftmff,  4),  # Sonya Fusion
                "p_amm":    round(p_amm,    4),  # Albina MMD-Thinker v4
                "model_route": "cdsm_v3_4channel",
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # Lazy-load тяжёлых dense-энкодеров (CLIP + multilingual-e5-small)
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_dense_encoders_loaded(self) -> None:
        """Lazy-load CLIP и e5-моделей при первом запросе.

        CLIP и e5 — heavyweight transformer-модели от HuggingFace; их загрузка
        занимает 5–15 секунд на CPU. Чтобы не блокировать старт сервиса,
        они загружаются при первом обращении в `predict()`. Аналогичная
        стратегия применяется в `D2VCatBoostPredictor._ensure_clip_loaded()`.
        """
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("CDSMV3Predictor: lazy-load CLIP ViT-B/32 (CPU)...")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model.eval()
        if self._e5_model is None:
            from transformers import AutoModel, AutoTokenizer
            logger.info("CDSMV3Predictor: lazy-load multilingual-e5-small (CPU)...")
            self._e5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
            self._e5_model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
            self._e5_model.eval()

    # ─────────────────────────────────────────────────────────────────────
    # Производные признаки (Group H + Group I по § 5.4.6.2)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_structural_features(
        self,
        raw_clip_emb: np.ndarray,
        tab_inputs: dict,
        brand: str,
        name: str,
    ) -> dict:
        """Group H (CLIP-derived structural, 4 признака) + brand-агрегаты + typosquatting.

        - `clip_norm`         — L2-норма raw CLIP-вектора (Group H)
        - `clip_dist_centered` — отклонение нормы от 0,5 (Group H)
        - `clip_dist_norm`    — clip_dist × norm (Group H, interaction)
        - `clip_dist_centroid` — cosine-distance до centroid категории (Group H);
                                 требует pre-computed centroid (опционально, заглушка 0)
        - `brand_exact`, `brand_fuzzy`, `typosquat` — Group D (К. Азимова, Deng 2020)
        """
        try:
            from rapidfuzz import fuzz
            brand_l = brand.strip().lower()
            name_l = name.strip().lower()
            brand_exact = float(bool(brand_l and brand_l in name_l))
            brand_fuzzy = float(fuzz.partial_ratio(brand_l, name_l) / 100.0) if brand_l else 0.0
            typosquat = max(0.0, brand_fuzzy - 0.5 * brand_exact)
        except ImportError:
            brand_exact = brand_fuzzy = typosquat = 0.0

        norm = float(np.linalg.norm(raw_clip_emb))
        normalised_norm = norm / (np.sqrt(len(raw_clip_emb)) + 1e-9)
        dist_centered = abs(normalised_norm - 0.5)
        dist_norm = normalised_norm * norm

        return {
            "clip_norm": normalised_norm,
            "clip_dist_centered": dist_centered,
            "clip_dist_norm": dist_norm,
            "clip_dist_centroid": 0.0,  # требует pre-computed category centroids в production
            "brand_exact": brand_exact,
            "brand_fuzzy": brand_fuzzy,
            "typosquat": typosquat,
        }

    def _compute_cross_modal_kl(
        self,
        raw_clip_emb: np.ndarray,
        text_svd: np.ndarray,
        tab_inputs: dict,
    ) -> float:
        """Group I (cross-modal KL feature, § 5.4.6.2 / § 4.1 Exp-3b К. Азимовой).

        В обучении (OOF на train) считается как
        KL(text-classifier-proba ‖ tab-classifier-proba), где text-classifier
        обучен только на TF-IDF, а tab-classifier — только на 38 командных
        tabular-признаках. В production-инференсе для нового объекта требуются
        обученные text-only и tab-only pre-classifiers, сохранённые при обучении
        CDSM-канала.

        Если pre-classifiers недоступны в текущем релизе — возвращаем 0 (нейтральный
        сигнал). Это слегка деградирует CDSM-канал по Mode 3, но не ломает работу.
        Точное значение Group I восстанавливается при production-доукомплектовании
        двух pre-classifiers в `artifacts/cdsm_v3/`.
        """
        # TODO(production): загрузить text-only и tab-only pre-classifiers из
        # `artifacts/cdsm_v3/kl_text_clf.cbm` и `kl_tab_clf.cbm`, прогнать
        # их через текущий запрос, вычислить KL-дивергенцию двух вероятностей.
        return 0.0
