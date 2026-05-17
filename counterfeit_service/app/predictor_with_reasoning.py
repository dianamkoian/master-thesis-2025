"""
ReasoningPredictor — predictor-agnostic wrapper, добавляющий LLM-канал
explanation поверх любого зарегистрированного primary predictor'а.

Архитектура (§ 4.4.9.7 ВКР):
  inner_predictor.predict(...)  →  (is_counterfeit, probability, signals)
       │
       ▼
  для borderline (probability ∈ (BORDERLINE_LO, BORDERLINE_HI))
       │
       ▼
  LLMExplainer.explain(card, probability, signals)
       │
       ▼
  расширенный контракт: { ..., "reasoning": str|None, "reasoning_mode": str }

Активация: переменная окружения `PREDICTOR_TYPE=reasoning_pipeline` плюс
`PREDICTOR_INNER=<inner-name>` (по умолчанию `d2v_catboost`).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .predictor_base import BasePredictor

logger = logging.getLogger(__name__)

DEFAULT_INNER = os.getenv("PREDICTOR_INNER", "d2v_catboost")
BORDERLINE_LO = float(os.getenv("REASONING_BORDERLINE_LO", "0.25"))
BORDERLINE_HI = float(os.getenv("REASONING_BORDERLINE_HI", "0.75"))
# Порог автоматической блокировки. Reasoning поверх этого порога
# предназначен для уведомления селлера / тикета в поддержку.
BLOCKING_THRESHOLD = float(os.getenv("REASONING_BLOCKING_THRESHOLD", "0.85"))
# По умолчанию используется детерминированный rule-based explainer
# (§ 4.4.9.7, обоснование в `template_reasoner.py`). LLM-канал на базе
# Qwen2.5 включается через USE_LLM_REASONING=1 — оставлен как опциональный
# research-канал для деплоев с MPS/GPU.
USE_LLM_REASONING = os.getenv("USE_LLM_REASONING", "0") in ("1", "true", "True", "yes")


class ReasoningPredictor(BasePredictor):
    """Wrapper: вердикт от inner predictor + reasoning от LLM на borderline."""

    name = "reasoning_pipeline"
    description = (
        "Caskаdная архитектура: вердикт от inner predictor "
        f"(default={DEFAULT_INNER}) + verdict-conditioned reasoning от "
        "Qwen2.5-1.5B-Instruct для borderline-объектов "
        f"(proba ∈ ({BORDERLINE_LO}, {BORDERLINE_HI}))."
    )

    def __init__(
        self,
        inner_name: str = DEFAULT_INNER,
        borderline_lo: float = BORDERLINE_LO,
        borderline_hi: float = BORDERLINE_HI,
        blocking_threshold: float = BLOCKING_THRESHOLD,
    ):
        self.inner_name = inner_name
        self.borderline_lo = borderline_lo
        self.borderline_hi = borderline_hi
        self.blocking_threshold = blocking_threshold
        self.inner: Optional[BasePredictor] = None
        self.llm = None  # LLMExplainer

    def load(self) -> None:
        """Загрузить inner predictor + (опционально) LLMExplainer.

        Reasoning-канал по умолчанию работает в template-based режиме
        (`template_reasoner.generate_reasoning`) — мгновенный, детерминированный,
        не зависит от GPU/MPS. LLM-канал (Qwen2.5-1.5B-Instruct) включается
        через `USE_LLM_REASONING=1`; для приемлемого latency требует MPS/CUDA.
        """
        # Поздний импорт, чтобы избежать кругового импорта с predictor.py.
        from .predictor import PREDICTOR_REGISTRY

        if self.inner_name not in PREDICTOR_REGISTRY:
            raise ValueError(
                f"Inner predictor {self.inner_name!r} не зарегистрирован в "
                f"PREDICTOR_REGISTRY. Доступны: {list(PREDICTOR_REGISTRY)}"
            )
        cls = PREDICTOR_REGISTRY[self.inner_name]
        if cls is ReasoningPredictor:
            raise ValueError("inner predictor не может быть reasoning_pipeline (циклическая ссылка)")

        self.inner = cls()
        self.inner.load()

        self.llm = None
        if USE_LLM_REASONING:
            from .llm_explainer import LLMExplainer
            self.llm = LLMExplainer()
            if os.getenv("REASONING_EAGER_LOAD", "1") not in ("0", "false", "False"):
                try:
                    self.llm.load()
                    logger.info("LLM weights pre-loaded at startup (eager mode)")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("LLM eager-load failed (%s); fall back to lazy: %s",
                                   type(exc).__name__, exc)
            logger.info("Reasoning channel: LLM (Qwen2.5) ENABLED")
        else:
            logger.info("Reasoning channel: rule-based template (default; "
                        "set USE_LLM_REASONING=1 для LLM-канала)")
        logger.info(
            "ReasoningPredictor ready: inner=%s, borderline=(%.2f, %.2f), blocking>=%.2f",
            self.inner_name, self.borderline_lo, self.borderline_hi, self.blocking_threshold,
        )

    def predict(
        self,
        image_bytes: bytes,
        name: str,
        description: str,
        brand: str,
        tab_inputs: dict,
    ) -> dict:
        assert self.inner is not None, "predictor not loaded — call load() first"
        result = self.inner.predict(image_bytes, name, description, brand, tab_inputs)
        proba = float(result.get("probability", 0.0))

        # Три режима, в которых reasoning имеет ценность:
        #   1. blocking_explanation — proba >= blocking_threshold (для уведомления
        #      селлера или тикета в поддержку при автоблокировке);
        #   2. borderline_explanation — proba ∈ (borderline_lo, borderline_hi)
        #      (для модератора, который вручную решает borderline-кейс);
        #   3. confident_positive_no_block — proba ∈ [borderline_hi, blocking_threshold)
        #      (опционально для контроля качества — высокая уверенность, но не
        #      авто-блок). По умолчанию reasoning тоже генерируется.
        # confident_negative (proba <= borderline_lo) — reasoning не нужен.
        if proba >= self.blocking_threshold:
            mode = "blocking_explanation"
            do_explain = True
        elif self.borderline_lo < proba < self.blocking_threshold:
            mode = ("borderline_explanation"
                    if proba < self.borderline_hi else "confident_positive_no_block")
            do_explain = True
        else:
            mode = "confident_negative"
            do_explain = False

        if do_explain:
            card = {
                "name": name,
                "description": description,
                "brand": brand,
                **(tab_inputs or {}),
            }
            try:
                if self.llm is not None:
                    reasoning = self.llm.explain(card, proba, result.get("signals"), mode=mode)
                else:
                    # По умолчанию — детерминированный rule-based explainer.
                    from .template_reasoner import generate_reasoning
                    reasoning = generate_reasoning(card, proba, mode=mode)
                result["reasoning"] = reasoning
                result["reasoning_mode"] = mode
            except Exception as e:  # noqa: BLE001
                logger.exception("Reasoning generation failed: %s", e)
                result["reasoning"] = None
                result["reasoning_mode"] = "reasoning_error"
        else:
            result["reasoning"] = None
            result["reasoning_mode"] = mode
        return result

    def health_info(self) -> dict:
        return {
            "predictor": self.name,
            "description": self.description,
            "inner_predictor": self.inner_name,
            "borderline_lo": self.borderline_lo,
            "borderline_hi": self.borderline_hi,
        }
