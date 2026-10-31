


from pydantic import BaseModel, Field


class feature(BaseModel):
    clean_text: str = Field(...)


class outlet(BaseModel):
    Station: str
    features: feature