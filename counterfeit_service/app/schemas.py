import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Signals(BaseModel):
    multimodal_score: float = Field(..., description="Full fusion model probability")
    image_signal: float = Field(..., description="Image-only contribution (img features, others zeroed)")
    text_signal: float = Field(..., description="Text-only contribution (d2v features, others zeroed)")


class PredictionResponse(BaseModel):
    is_counterfeit: bool
    probability: float
    signals: Signals


class PredictionRecord(BaseModel):
    id: uuid.UUID
    created_at: datetime
    seller_id: Optional[str]
    product_name: Optional[str]
    brand: Optional[str]
    category: Optional[str]
    is_counterfeit: bool
    probability: float
    signals: Signals

    model_config = {"from_attributes": True}


class FeedbackCreate(BaseModel):
    correct: bool = Field(..., description="Was the model prediction correct?")
    true_label: bool = Field(..., description="Actual counterfeit label (ground truth)")
    moderator_comment: Optional[str] = None
    moderator_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: uuid.UUID
    prediction_id: uuid.UUID
    created_at: datetime
    correct: bool
    true_label: bool
    moderator_comment: Optional[str]
    moderator_id: Optional[str]

    model_config = {"from_attributes": True}


class SellerProfileResponse(BaseModel):
    seller_id: str
    first_seen: datetime
    last_seen: datetime
    total_predictions: int
    flagged_count: int
    avg_probability: float
    confirmed_counterfeit_count: int

    model_config = {"from_attributes": True}

