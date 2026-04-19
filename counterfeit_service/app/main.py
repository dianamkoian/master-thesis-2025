import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, init_db
from app.models import Feedback, Prediction, SellerProfile
from app.predictor import CounterfeitPredictor
from app.schemas import (
    FeedbackCreate,
    FeedbackResponse,
    PredictionRecord,
    PredictionResponse,
    SellerProfileResponse,
)

_HERE = Path(__file__).parent.parent
STATIC_DIR = str(os.getenv("STATIC_DIR", _HERE / "static"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

predictor = CounterfeitPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    await init_db()
    yield


app = FastAPI(
    title="Counterfeit Detection Service",
    description="Multimodal counterfeit product detection for Ozon marketplace",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(Path(STATIC_DIR) / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Product photo"),
    name: str = Form("", description="Product name (name_rus)"),
    description: str = Form("", description="Product description"),
    brand: str = Form("", description="Brand name"),
    category: str = Form("", description="CommercialTypeName4 — product category string"),
    price: float = Form(0.0, description="PriceDiscounted"),
    item_time_alive: float = Form(0.0, description="Days on marketplace"),
    item_count_sales30: float = Form(0.0, description="Sales last 30 days"),
    item_count_returns30: float = Form(0.0, description="Returns last 30 days"),
    seller_time_alive: float = Form(0.0, description="Seller age in days"),
    seller_id: Optional[str] = Form(None, description="Seller identifier"),
    db: AsyncSession = Depends(get_db),
):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    tab_inputs = {
        "CommercialTypeName4": category,
        "PriceDiscounted": price,
        "item_time_alive": item_time_alive,
        "item_count_sales30": item_count_sales30,
        "item_count_returns30": item_count_returns30,
        "seller_time_alive": seller_time_alive,
    }

    try:
        result = predictor.predict(
            image_bytes=image_bytes,
            name=name,
            description=description,
            brand=brand,
            tab_inputs=tab_inputs,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    prediction = Prediction(
        seller_id=seller_id,
        product_name=name,
        description=description,
        brand=brand,
        category=category,
        price=price,
        item_time_alive=item_time_alive,
        item_count_sales30=item_count_sales30,
        item_count_returns30=item_count_returns30,
        seller_time_alive=seller_time_alive,
        image_filename=image.filename,
        is_counterfeit=result["is_counterfeit"],
        probability=result["probability"],
        multimodal_score=result["signals"]["multimodal_score"],
        image_signal=result["signals"]["image_signal"],
        text_signal=result["signals"]["text_signal"],
    )
    db.add(prediction)

    if seller_id:
        profile = await db.get(SellerProfile, seller_id)
        if profile is None:
            profile = SellerProfile(
                seller_id=seller_id,
                total_predictions=0,
                flagged_count=0,
                avg_probability=0.0,
                confirmed_counterfeit_count=0,
            )
            db.add(profile)
        new_total = (profile.total_predictions or 0) + 1
        profile.avg_probability = (
            (profile.avg_probability or 0.0) * (profile.total_predictions or 0) + result["probability"]
        ) / new_total
        profile.total_predictions = new_total
        profile.last_seen = datetime.utcnow()
        if result["is_counterfeit"]:
            profile.flagged_count += 1

    await db.commit()

    return PredictionResponse(**result)


@app.post("/predictions/{prediction_id}/feedback", response_model=FeedbackResponse)
async def add_feedback(
    prediction_id: uuid.UUID,
    body: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    prediction = await db.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    fb = Feedback(prediction_id=prediction_id, **body.model_dump())
    db.add(fb)

    if prediction.seller_id and body.true_label:
        profile = await db.get(SellerProfile, prediction.seller_id)
        if profile:
            profile.confirmed_counterfeit_count += 1

    await db.commit()
    await db.refresh(fb)
    return fb


@app.get("/predictions", response_model=list[PredictionRecord])
async def list_predictions(
    is_counterfeit: Optional[bool] = None,
    seller_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit).offset(offset)
    if is_counterfeit is not None:
        stmt = stmt.where(Prediction.is_counterfeit == is_counterfeit)
    if seller_id:
        stmt = stmt.where(Prediction.seller_id == seller_id)
    rows = await db.execute(stmt)
    predictions = rows.scalars().all()
    return [
        PredictionRecord(
            id=p.id,
            created_at=p.created_at,
            seller_id=p.seller_id,
            product_name=p.product_name,
            brand=p.brand,
            category=p.category,
            is_counterfeit=p.is_counterfeit,
            probability=p.probability,
            signals={
                "multimodal_score": p.multimodal_score,
                "image_signal": p.image_signal,
                "text_signal": p.text_signal,
            },
        )
        for p in predictions
    ]


@app.get("/seller-profiles/{seller_id}", response_model=SellerProfileResponse)
async def get_seller_profile(seller_id: str, db: AsyncSession = Depends(get_db)):
    profile = await db.get(SellerProfile, seller_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Seller not found")
    return profile
