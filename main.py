from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import datetime


SQLALCHEMY_DATABASE_URL = "sqlite:///./ml_logs.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})


#make session 

sessionLocal = sessionmaker(autocommit= False, autoflush = False, bind= engine )




Base = declarative_base() # using the base class 

class PredictionLog(Base):

    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key = True, index= True)

    # input column 

    input_text = Column(String)

    #prediction column

    prediction_text = Column(String)

    # timestamp column

    timestamp = Column(DateTime, default= datetime.datetime.utc.now)

    Base.metadata.create_all(bind= engine )



def get_db():

    db = sessionLocal()


    try:

        yield db
    finally:

        db.close()




app = FASTAPI()

class PredictionRequest(BaseModel):
    input_text: str

@app.post("/predict")

async def make_prediction(request: PredictionRequest, db: Session = Depends(get_db)):

    dummy_prediction = len(request.input_text) *1.5

    new_log = PredictionLog(
        input_text=request.input_text, predicted_value=dummy_prediction
    )

    db.add(new_log)

    db.commit()


    db.refresh(new_log)

    return {
        
        "status": "success",
        "log_id": new_log.id,
        "input": request.input_text,
        "prediction": dummy_prediction,
    }





