"""
Конкретная реализация `BasePredictor` для command-v1 модели сервиса
детекции контрафакта (Feature Fusion на CatBoost с Doc2Vec и CLIP).

Модель ожидает столбцы в этом порядке (подтверждено из `model.feature_names_`):
  [0:38]   табличные признаки (feature_cols, CommercialTypeName4 — категориальная строка)
  [38:238] Doc2Vec 200-мерные эмбеддинги (d2v_0 .. d2v_199), без масштабирования
  [238:750] CLIP-эмбеддинги изображения (img_0 .. img_511), отмасштабированные

Файл `d2v_model.pkl` должен лежать в `artifacts/` (см.
`counterfeit_service/scripts/retrain_d2v_model.py` для пересохранения в формате,
совместимом с современной numpy). При отсутствии файла d2v_-столбцы заполняются
нулями с логированием warning — модель остаётся работоспособной только на
табличной и визуальной модальностях.

Эта реализация **deprecated в финальной headline-конфигурации работы** (§ 4.4.3,
§ 4.4.8 фиксирует Doc2Vec как Negative Transfer 1 относительно TF-IDF SVD), но
сохраняется как baseline-артефакт соревнования Ozon eCup 2025. Для подключения
финальных M2-FE+ или ансамбля см. `counterfeit_service/MODELS.md`.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from PIL import Image
import io

from app.predictor_base import BasePredictor

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent.parent  # counterfeit_service/
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", _HERE / "artifacts"))

# Fixed column counts verified from model.feature_names_
N_D2V = 200
N_IMG = 512
D2V_COLS = [f"d2v_{i}" for i in range(N_D2V)]
IMG_COLS = [f"img_{i}" for i in range(N_IMG)]

# Prediction threshold for is_counterfeit verdict
THRESHOLD = float(os.getenv("COUNTERFEIT_THRESHOLD", "0.5"))


def _apply_legacy_numpy_pickle_compat():
    """Восстанавливает совместимость с pickle, сохранённым в numpy <1.17,
    где BitGenerator передавался в __bit_generator_ctor как класс, а не имя.
    Без этого joblib.load(d2v_model.pkl) падает с ValueError на современной numpy."""
    import numpy.random._pickle as _np_pickle
    attr = "_bit_generator_ctor" if hasattr(_np_pickle, "_bit_generator_ctor") else "__bit_generator_ctor"
    orig = getattr(_np_pickle, attr, None)
    if orig is None or getattr(orig, "_legacy_patched", False):
        return
    def patched(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return orig(bit_generator_name)
    patched._legacy_patched = True
    setattr(_np_pickle, attr, patched)


class D2VCatBoostPredictor(BasePredictor):
    """Command-v1 модель сервиса: Doc2Vec + CLIP + CatBoost (Feature Fusion).

    См. модульный docstring для описания контракта столбцов и расположения
    артефактов. Реализация соблюдает контракт `BasePredictor`.
    """

    name = "d2v_catboost"
    description = (
        "Command-v1 baseline: 38 tab + Doc2Vec(200) + CLIP(512) → CatBoost. "
        "Deprecated в финальной headline-конфигурации (см. § 4.4.3.3)."
    )

    def __init__(self):
        self.model: Optional[CatBoostClassifier] = None
        self.img_scaler = None
        self.feature_cols: list = []
        self.cat_cols: list = []
        self.d2v_model = None
        self._clip_model = None
        self._clip_processor = None
        self._clip_loaded = False

    def load(self):
        logger.info("Loading artifacts from %s", ARTIFACTS_DIR)

        # CatBoost model
        self.model = CatBoostClassifier()
        self.model.load_model(str(ARTIFACTS_DIR / "catboost_model.cbm"))
        logger.info("CatBoost loaded: %d features", len(self.model.feature_names_))

        # Tabular metadata
        self.feature_cols = joblib.load(ARTIFACTS_DIR / "feature_cols.pkl")
        self.cat_cols = joblib.load(ARTIFACTS_DIR / "cat_cols.pkl")
        logger.info("feature_cols: %d, cat_cols: %s", len(self.feature_cols), self.cat_cols)

        # Image scaler
        self.img_scaler = joblib.load(ARTIFACTS_DIR / "img_scaler.pkl")
        logger.info("img_scaler loaded, expects %d dims", self.img_scaler.n_features_in_)

        # Doc2Vec model (optional — saved from training notebook).
        # Graceful degradation: при несовместимости pickle с текущей numpy /
        # gensim (`MT19937 is not a known BitGenerator module` и подобные)
        # text-modality заполняется нулями, image и tabular модальности
        # продолжают работать в полной мере. См. § 7.5 ВКР: «отказоустойчивость
        # к деградации артефактов».
        d2v_path = ARTIFACTS_DIR / "d2v_model.pkl"
        if d2v_path.exists():
            _apply_legacy_numpy_pickle_compat()
            try:
                self.d2v_model = joblib.load(d2v_path)
                logger.info("Doc2Vec model loaded")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "d2v_model.pkl не загружен (%s: %s). "
                    "Текстовая модальность будет заполнена нулями; image и "
                    "tabular работают штатно. Чтобы восстановить: пересохранить "
                    "артефакт через `python save_d2v_model.py --retrain` в той же "
                    "версии numpy/gensim, что развёрнута в сервисе.",
                    type(exc).__name__, exc,
                )
                self.d2v_model = None
        else:
            logger.warning(
                "d2v_model.pkl not found in artifacts/. "
                "Text modality (d2v_0..d2v_199) will be filled with zeros. "
                "Run save_d2v_model.py after training to save it."
            )

        # CLIP is loaded lazily on first image request
        logger.info("Predictor ready (CLIP will load on first request)")

    def _load_clip(self):
        if self._clip_loaded:
            return
        logger.info("Loading CLIP ViT-B/32 model (CPU)...")
        from transformers import CLIPModel, CLIPProcessor
        self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model.eval()
        self._clip_loaded = True
        logger.info("CLIP loaded")

    def _get_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """PIL image bytes → CLIP 512-dim → img_scaler → array[512]"""
        import torch
        self._load_clip()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._clip_processor(images=image, return_tensors="pt")
        # В новых версиях transformers `get_image_features` иногда возвращает
        # неправильно-формованный тензор для batch=1 (см. baseline-bug
        # § 7.5.2 ВКР). Явный pipeline `vision_model.pooler_output` +
        # `visual_projection` детерминирован и даёт ровно 512-dim вектор.
        with torch.no_grad():
            vision_outputs = self._clip_model.vision_model(pixel_values=inputs.pixel_values)
            pooled = vision_outputs.pooler_output                      # (1, hidden_size)
            projected = self._clip_model.visual_projection(pooled)     # (1, 512)
        embedding = projected[0].numpy().astype(np.float32)
        assert embedding.shape == (512,), (
            f"CLIP-embedding contract violation: expected (512,), got {embedding.shape}"
        )
        scaled = self.img_scaler.transform(embedding.reshape(1, -1))[0]
        return scaled  # (512,)

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """text → Doc2Vec infer_vector(200,) or zeros if model not available"""
        if self.d2v_model is not None:
            tokens = text.lower().split()
            if not tokens:
                return np.zeros(N_D2V, dtype=np.float32)
            vec = self.d2v_model.infer_vector(tokens, epochs=50)
            return np.array(vec, dtype=np.float32)
        return np.zeros(N_D2V, dtype=np.float32)

    def _build_tabular_row(self, tab_inputs: dict) -> pd.DataFrame:
        """
        Build a single-row DataFrame with all feature_cols.
        Missing numeric columns are filled with 0.0.
        CommercialTypeName4 stays as string.
        """
        row = {}
        for col in self.feature_cols:
            if col in self.cat_cols:
                val = tab_inputs.get(col)
                row[col] = str(val) if val not in (None, "") else ""
            else:
                val = tab_inputs.get(col)
                row[col] = float(val) if val not in (None, "") else 0.0
        return pd.DataFrame([row])[self.feature_cols]

    def _build_fused_df(
        self,
        tab_inputs: dict,
        d2v_vec: np.ndarray,
        img_vec: np.ndarray,
    ) -> pd.DataFrame:
        """Concatenate all modalities in training order: tabular → d2v → img"""
        df_tab = self._build_tabular_row(tab_inputs)
        df_d2v = pd.DataFrame([d2v_vec], columns=D2V_COLS)
        df_img = pd.DataFrame([img_vec], columns=IMG_COLS)
        fused = pd.concat([df_tab, df_d2v, df_img], axis=1)
        return fused

    def _predict_proba(self, fused_df: pd.DataFrame) -> float:
        prob = self.model.predict_proba(fused_df)[0][1]
        return float(prob)

    def predict(
        self,
        image_bytes: bytes,
        name: str,
        description: str,
        brand: str,
        tab_inputs: dict,
    ) -> dict:
        """
        Full inference pipeline.

        Returns:
            is_counterfeit, probability, signals (multimodal, image, text)
        """
        # Build combined text field same as training:
        # train_df['text'] = name_rus + ' ' + description + ' ' + brand_name
        text = f"{name} {description} {brand}".strip()

        # Get embeddings
        img_vec = self._get_image_embedding(image_bytes)   # (512,) scaled
        d2v_vec = self._get_text_embedding(text)           # (200,) unscaled

        # ── Main prediction (all modalities) ──
        fused = self._build_fused_df(tab_inputs, d2v_vec, img_vec)
        multimodal_score = self._predict_proba(fused)

        # ── Image-only signal: zero out d2v, keep img ──
        zeros_d2v = np.zeros(N_D2V, dtype=np.float32)
        tab_zero = {col: "" if col in self.cat_cols else 0.0 for col in self.feature_cols}
        tab_zero["CommercialTypeName4"] = str(tab_inputs.get("CommercialTypeName4", ""))
        fused_img_only = self._build_fused_df(tab_zero, zeros_d2v, img_vec)
        image_signal = self._predict_proba(fused_img_only)

        # ── Text-only signal: zero out img, keep d2v ──
        zeros_img = np.zeros(N_IMG, dtype=np.float32)
        fused_text_only = self._build_fused_df(tab_zero, d2v_vec, zeros_img)
        text_signal = self._predict_proba(fused_text_only)

        return {
            "is_counterfeit": multimodal_score >= THRESHOLD,
            "probability": round(multimodal_score, 4),
            "signals": {
                "multimodal_score": round(multimodal_score, 4),
                "image_signal": round(image_signal, 4),
                "text_signal": round(text_signal, 4),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Адаптивный выбор предиктора через env `PREDICTOR_TYPE`
#
# Сервис задуман как pluggable: чтобы подключить новую модель, нужно
# реализовать `BasePredictor` (см. `app/predictor_base.py`) и зарегистрировать
# свой класс в словаре `PREDICTOR_REGISTRY` ниже. После этого выставление
# `PREDICTOR_TYPE=<key>` в окружении выберет вашу модель без изменений
# `main.py` или `worker.py`. Пошаговая инструкция — `MODELS.md`.
# ─────────────────────────────────────────────────────────────────────────────

from .predictor_with_reasoning import ReasoningPredictor
from .predictor_stub import StubBorderlinePredictor
from .predictor_cdsm import CDSMV3Predictor
from .predictor_cdsm_headline import CDSMV3HeadlinePredictor
from .predictor_hcdm import HCDMHeadlinePredictor

PREDICTOR_REGISTRY: dict[str, type[BasePredictor]] = {
    "hcdm_4channel": HCDMHeadlinePredictor,          # Wolpert-валидированный headline § 5.4.X / § 5.6 ВКР: HCDM — Иерархическая кросс-доменная модель (PR=0.8044, R@P=0.2068, ROC=0.9720 — итог Гл.5 ВКР)
    "cdsm_v3_4channel": CDSMV3HeadlinePredictor,     # historical headline production-канал § 5.4.6 ВКР: frozen probas baseline_v3_4ch (C=0.1, PR=0.7579, R@P=0.2078, ROC=0.9603 — Phase 1 итог, см. § 5.4.5–5.4.6)
    "cdsm_v3_4channel_live": CDSMV3Predictor,        # экспериментальный live-инференс через 4 раздельные модели (требует артефактов соавторов)
    "d2v_catboost": D2VCatBoostPredictor,            # baseline для отладки и регрессионного тестирования
    "reasoning_pipeline": ReasoningPredictor,        # § 4.4.9.7: verdict + LLM reasoning
    "stub_borderline": StubBorderlinePredictor,      # демо-stub для проверки LLM-канала
    # "lodo_precision_3channel": LODOPrecisionPredictor,  # TODO: production-канал автоматических действий, § 5.6.2
    # "m2_fe_plus": M2FeaturePlusPredictor,                # TODO: standalone Дианина headline M2-FE+
}


def get_predictor(model_type: Optional[str] = None) -> BasePredictor:
    """Вернуть инстанс предиктора по типу.

    Источник имени:
      1. Аргумент `model_type`, если передан явно.
      2. Переменная окружения `PREDICTOR_TYPE`.
      3. По умолчанию — `cdsm_v3_4channel` (headline production-канал ранжирования
         согласно § 5.6.2 ВКР; при отсутствии артефактов соавторов выполняется
         graceful fallback на `d2v_catboost` baseline с предупреждением в /health).
    """
    name = model_type or os.getenv("PREDICTOR_TYPE", "cdsm_v3_4channel")
    if name not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown PREDICTOR_TYPE='{name}'. Доступные: {list(PREDICTOR_REGISTRY)}. "
            f"Зарегистрируйте свою модель в PREDICTOR_REGISTRY (см. MODELS.md)."
        )
    logger.info("Selected predictor: %s", name)
    return PREDICTOR_REGISTRY[name]()


# Обратная совместимость: `main.py` и `worker.py` импортируют `CounterfeitPredictor`.
# Сохраняем имя как alias до окончательного перехода на `get_predictor()`.
CounterfeitPredictor = D2VCatBoostPredictor
