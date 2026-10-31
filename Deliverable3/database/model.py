from sqlalchemy import Column, Integer, String, Float, Date
from Deliverable3.database.database import Base

class News_text(Base):
    __tablename__ = "News"
    News_code = Column(Integer, primary_key=True, index = True)
    Station = Column(String)
    clean_text = Column(String)
    prediction = Column(String)
