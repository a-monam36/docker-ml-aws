import datetime
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import joblib
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("property_valuation_service")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./ml_logs.db"

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    sqft = Column(Float, nullable=False)
    bedrooms = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    predicted_price = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


# Global model reference
model = None


def load_model_artifact():
    """Load model artifact from AWS S3 if configured, falling back to local file."""
    global model
    s3_bucket = os.getenv("S3_BUCKET_NAME")

    if s3_bucket:
        try:
            logger.info("S3_BUCKET_NAME is set to '%s'. Attempting download...", s3_bucket)
            import boto3
            s3 = boto3.client("s3")
            s3.download_file(s3_bucket, "model.joblib", "model.joblib")
            logger.info("Successfully downloaded model.joblib from S3 bucket '%s'.", s3_bucket)
        except Exception as exc:
            logger.warning(
                "Failed to download model from S3 (%s). Falling back to local model.joblib.",
                exc
            )

    try:
        if os.path.exists("model.joblib"):
            model = joblib.load("model.joblib")
            logger.info("Successfully loaded local model artifact 'model.joblib'.")
        else:
            logger.warning("Artifact 'model.joblib' not found. Model is uninitialized.")
            model = None
    except Exception as exc:
        logger.error("Failed to deserialize 'model.joblib': %s", exc)
        model = None


# Initial safe table creation and model load
Base.metadata.create_all(bind=engine)
load_model_artifact()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown events."""
    # Ensure tables exist on startup
    Base.metadata.create_all(bind=engine)
    # Safe model initialization on startup
    load_model_artifact()
    yield


app = FastAPI(
    title="Real Estate Property Valuation Microservice",
    description="Microservice providing property valuation predictions with SQLite/AWS logging.",
    version="1.0.0",
    lifespan=lifespan,
)


class PropertyFeatures(BaseModel):
    sqft: float = Field(..., gt=0, description="Square footage must be strictly greater than 0")
    bedrooms: int = Field(..., ge=0, description="Bedrooms count must be non-negative (>= 0)")
    age: int = Field(..., ge=0, description="Property age in years must be non-negative (>= 0)")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def health_check():
    """Health check endpoint exposing service health and model status."""
    return {
        "status": "healthy",
        "service": "Real Estate Property Valuation API",
        "model_loaded": model is not None,
        "database": "connected"
    }


@app.post("/predict")
def predict(features: PropertyFeatures, db: Session = Depends(get_db)):
    """Generate valuation prediction, record latency, and save audit record."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Train model or verify model.joblib exists."
        )

    start_time = time.perf_counter()
    try:
        prediction = float(model.predict([[features.sqft, features.bedrooms, features.age]])[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}"
        )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    new_log = PredictionLog(
        sqft=features.sqft,
        bedrooms=features.bedrooms,
        age=features.age,
        predicted_price=round(prediction, 2),
        latency_ms=round(latency_ms, 2),
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)

    return {
        "status": "success",
        "log_id": new_log.id,
        "prediction": round(prediction, 2),
        "predicted_price": round(prediction, 2),
        "latency_ms": round(latency_ms, 2),
        "features": {
            "sqft": features.sqft,
            "bedrooms": features.bedrooms,
            "age": features.age,
        },
        "timestamp": new_log.timestamp.isoformat() if new_log.timestamp else None
    }


@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    """Retrieve the 10 most recent prediction logs for auditing."""
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.id.desc())
        .limit(10)
        .all()
    )
    return [
        {
            "id": log.id,
            "sqft": log.sqft,
            "bedrooms": log.bedrooms,
            "age": log.age,
            "predicted_price": log.predicted_price,
            "latency_ms": log.latency_ms,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
        }
        for log in logs
    ]
