from typing import Optional

from pydantic import BaseModel, Field


class Signals(BaseModel):
    multimodal_score: float = Field(..., description="Full fusion model probability")
    image_signal: Optional[float] = Field(
        default=None,
        description="Image-only contribution (img features, others zeroed). "
                    "Для HCDM frozen-lookup равно multimodal_score (headline-проба § 5.4.6 ВКР). "
                    "Для HCDM live режима возвращается None; раздельная image/text-декомпозиция "
                    "требует дополнительных проходов 4-канальной композиции с маскированными "
                    "модальностями — вместо них клиент использует канал-уровневые вклады "
                    "p_social/p_mobile_image/p_realestate/p_fintech ниже.",
    )
    text_signal: Optional[float] = Field(
        default=None,
        description="Text-only contribution (text features, others zeroed). См. image_signal: "
                    "для HCDM live режима возвращается None, эквивалентная декомпозиция — через канал-уровневые p_*.",
    )
    model_route: Optional[str] = Field(
        default=None,
        description="Какой канал отдал ответ: hcdm_4channel_headline | hcdm_4channel_live | "
                    "cdsm_v3_4channel_headline | hcdm_headline_fallback_baseline | другие. "
                    "Помогает в защите ВКР показать, что headline-числа § 5.4.6 ВКР приходят "
                    "именно из канонической lookup-таблицы, а live-инференс — из 4-канальной "
                    "композиции с правильными весами.",
    )
    # Channel-level вероятности HCDM для live-режима (§ 5.4.6.3 ВКР).
    # Заполняются только при `model_route == "hcdm_4channel_live"`, чтобы пользователь
    # видел декомпозицию ансамбля. Для frozen-lookup и baseline-fallback остаются None.
    p_social: Optional[float] = Field(
        default=None,
        description="Канал-якорь HCDM: Mode 3 (Deep Deliberation) MMD-Thinker (соц. сети, w=0,875)",
    )
    p_mobile_image: Optional[float] = Field(
        default=None,
        description="Ortho diversifier HCDM: Baseline C image-only LogReg (мобильные приложения, w=0,075)",
    )
    p_realestate: Optional[float] = Field(
        default=None,
        description="Канал HCDM: M2-FE+ CatBoost с Group D Deng integration (недвижимость, w=0,025)",
    )
    p_fintech: Optional[float] = Field(
        default=None,
        description="Канал HCDM: FT-MFF late fusion (финтех, w=0,025)",
    )


class PredictionResponse(BaseModel):
    """Синхронный ответ /predict — результат вычисляется сразу."""
    is_counterfeit: bool
    probability: float
    signals: Signals
    # § 4.4.9.7 ВКР: verdict-conditioned reasoning. Заполняется predictor'ом
    # `reasoning_pipeline`; для других predictor'ов остаются None.
    reasoning: Optional[str] = Field(
        default=None,
        description="LLM-объяснение для borderline-вердиктов (заполнено только "
                    "при PREDICTOR_TYPE=reasoning_pipeline и proba ∈ (0.25, 0.75))",
    )
    reasoning_mode: Optional[str] = Field(
        default=None,
        description="Режим: borderline_explanation | confident_negative | "
                    "confident_positive | llm_error | pending",
    )
    # Progressive disclosure (§ 7.4). При запросе с `?defer_reasoning=true`
    # сервис возвращает быстрый verdict, генерация reasoning'а уходит в
    # BackgroundTask, клиент опрашивает GET /predict/{task_id}/reasoning.
    task_id: Optional[str] = Field(
        default=None,
        description="Идентификатор фоновой задачи генерации reasoning. "
                    "Непуст только при defer_reasoning=true.",
    )


class ReasoningResponse(BaseModel):
    """Ответ GET /predict/{task_id}/reasoning."""
    task_id: str
    status: str = Field(..., description="pending | ready | error | not_found")
    reasoning: Optional[str] = None
    reasoning_mode: Optional[str] = None
    error: Optional[str] = None


class TaskQueuedResponse(BaseModel):
    """Ответ /predict-async — задача поставлена в очередь RabbitMQ."""
    task_id: str = Field(..., description="UUID опубликованной задачи")
    status: str = Field(default="queued", description="queued | processing | done | error")


class AsyncResultResponse(BaseModel):
    """Ответ /result/{task_id} — статус и результат асинхронной задачи."""
    task_id: str
    status: str = Field(..., description="processing | done | error")
    is_counterfeit: Optional[bool] = None
    probability: Optional[float] = None
    signals: Optional[Signals] = None
    reasoning: Optional[str] = None
    reasoning_mode: Optional[str] = None
    error: Optional[str] = None
