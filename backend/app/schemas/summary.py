from pydantic import BaseModel , Field
from typing import Optional
from uuid import UUID
import datetime as dt

class SummaryCreate(BaseModel):
    original_text: Optional[str] = None
    summary_text: str

class SummaryView(BaseModel):
    id: UUID
    user_id: str
    original_text: Optional[str] = None
    summary_text: str
    created_at: dt.datetime

    class Config:
        from_attributes = True

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

class PredictOut(BaseModel):
    summary_text: str
    id: Optional[str] = None