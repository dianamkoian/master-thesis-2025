"""
StubBorderlinePredictor — детерминированный stub-predictor для демонстрации
LLM-канала без зависимости от CatBoost-артефактов и CLIP-инференса.

Возвращает фиксированный набор вероятностей в borderline-зоне для иллюстрации
работы `ReasoningPredictor`. Не предназначен для production. Активация:
`PREDICTOR_TYPE=stub_borderline` или `PREDICTOR_INNER=stub_borderline` внутри
`reasoning_pipeline`.

Probability выбирается по простой эвристике на основе age продавца и цены —
просто чтобы каждая тестовая карточка получала свой proba и LLM имел разный
input. Никакого ML здесь не происходит.
"""
from __future__ import annotations

import logging

from .predictor_base import BasePredictor

logger = logging.getLogger(__name__)


class StubBorderlinePredictor(BasePredictor):
    name = "stub_borderline"
    description = (
        "Демо-stub: возвращает proba ≈ 0.3–0.65 на основе возраста продавца и "
        "цены. Не использует ML, нужен для проверки LLM-канала без артефактов."
    )

    def load(self) -> None:
        logger.info("StubBorderlinePredictor: load() — no artefacts to load.")

    def predict(
        self,
        image_bytes: bytes,
        name: str,
        description: str,
        brand: str,
        tab_inputs: dict,
    ) -> dict:
        seller_age = float(tab_inputs.get("seller_time_alive", 30.0) or 30.0)
        price = float(tab_inputs.get("PriceDiscounted", 500.0) or 500.0)
        sales30 = float(tab_inputs.get("item_count_sales30", 0.0) or 0.0)
        returns30 = float(tab_inputs.get("item_count_returns30", 0.0) or 0.0)
        # Эвристика, специально подобранная под demo:
        # молодой продавец + низкая цена брендового товара → блок;
        # старый продавец + высокий объём продаж → разрешено;
        # средние сигналы → borderline.
        score = 0.5
        if seller_age < 30:
            score += 0.18
        elif seller_age > 365:
            score -= 0.25
        if brand.strip() and price < 15000:
            score += 0.25  # подозрительно дёшево для брендового
        elif price > 20000:
            score -= 0.10
        if returns30 > 0 and sales30 < 5:
            score += 0.05
        if not brand.strip() and seller_age > 365:
            score -= 0.10
        score = max(0.03, min(0.95, score))

        text_signal = 0.5 + (0.05 if not brand.strip() else -0.05)
        image_signal = 0.5
        return {
            "is_counterfeit": bool(score >= 0.5),
            "probability": round(score, 4),
            "signals": {
                "multimodal_score": round(score, 4),
                "image_signal": round(image_signal, 4),
                "text_signal": round(text_signal, 4),
            },
        }
