from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import datetime
import joblib
import os
import boto3

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ml_logs.db")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", None)

if "sqlite" in SQLALCHEMY_DATABASE_URL:
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    sqft = Column(Float)
    bedrooms = Column(Integer)
    age = Column(Integer)
    predicted_price = Column(Float)
    latency_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Downloads the model from S3 if a bucket name is provided via AWS EC2
if S3_BUCKET_NAME:
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET_NAME, 'model.joblib', 'model.joblib')

try:
    model = joblib.load("model.joblib")
except FileNotFoundError:
    model = None

class PropertyFeatures(BaseModel):
    sqft: float
    bedrooms: int
    age: int

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict")
def make_prediction(features: PropertyFeatures, db: Session = Depends(get_db)):
    start_time = datetime.datetime.utcnow()
    
    if model:
        prediction = model.predict([[features.sqft, features.bedrooms, features.age]])[0]
    else:
        prediction = 0.0

    end_time = datetime.datetime.utcnow()
    latency = (end_time - start_time).total_seconds() * 1000

    new_log = PredictionLog(
        sqft=features.sqft,
        bedrooms=features.bedrooms,
        age=features.age,
        predicted_price=float(prediction),
        latency_ms=latency
    )
    
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    
    return {
        "status": "success",
        "log_id": new_log.id,
        "prediction": round(prediction, 2),
        "latency_ms": round(latency, 2)
    }