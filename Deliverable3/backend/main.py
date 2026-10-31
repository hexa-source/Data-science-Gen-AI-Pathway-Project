
import uvicorn
from fastapi import FastAPI, Query, Depends
from Deliverable3.backend.schemas import feature
from Deliverable3.backend.schemas import outlet
from Deliverable3.backend.predictor import predict_news
from sqlalchemy.orm import Session
from Deliverable3.database.model import News_text
from Deliverable3.database.database import engine, Base, get_db 

Base.metadata.create_all(bind = engine)

app = FastAPI(title = "Fake News Detection System API", version = "2.0")


@app.post("/classify")
async def classify(News_code: int, Station: str, features: feature, db:Session = Depends(get_db)):
    prediction = predict_news(features.model_dump())
    segment = prediction["Prediction"]
    new_feature = News_text(News_code = News_code,
                            Station = Station,
                            clean_text = features.clean_text,
                            
                            prediction = segment)

    db.add(new_feature)
    db.commit()
    db.refresh(new_feature)

    return prediction
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
