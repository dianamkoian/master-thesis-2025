import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    seller_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    product_name: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    brand: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    price: Mapped[float | None] = mapped_column(Float)
    item_time_alive: Mapped[float | None] = mapped_column(Float)
    item_count_sales30: Mapped[float | None] = mapped_column(Float)
    item_count_returns30: Mapped[float | None] = mapped_column(Float)
    seller_time_alive: Mapped[float | None] = mapped_column(Float)
    image_filename: Mapped[str | None] = mapped_column(Text)

    is_counterfeit: Mapped[bool] = mapped_column(Boolean)
    probability: Mapped[float] = mapped_column(Float)
    multimodal_score: Mapped[float] = mapped_column(Float)
    image_signal: Mapped[float] = mapped_column(Float)
    text_signal: Mapped[float] = mapped_column(Float)

    feedback: Mapped[list["Feedback"]] = relationship(back_populates="prediction")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("predictions.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    correct: Mapped[bool] = mapped_column(Boolean)
    true_label: Mapped[bool] = mapped_column(Boolean)
    moderator_comment: Mapped[str | None] = mapped_column(Text)
    moderator_id: Mapped[str | None] = mapped_column(Text)

    prediction: Mapped["Prediction"] = relationship(back_populates="feedback")


class SellerProfile(Base):
    __tablename__ = "seller_profiles"

    seller_id: Mapped[str] = mapped_column(Text, primary_key=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    total_predictions: Mapped[int] = mapped_column(Integer, default=0)
    flagged_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_probability: Mapped[float] = mapped_column(Float, default=0.0)
    confirmed_counterfeit_count: Mapped[int] = mapped_column(Integer, default=0)
