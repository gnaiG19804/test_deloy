from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from uuid import UUID
import uuid, datetime as dt
from app.db.session import get_db
from app.core.config import settings
from app.models.summary import Summary
from app.schemas.summary import SummaryCreate,SummaryView, PredictIn
from app.core.security import get_current_user
from app.models.user import User
from sqlalchemy import desc
import requests
from app.llm.summarizer import summarize, load_model

router = APIRouter(prefix="/summaries", tags=["summaries"])

URL_PREDICT = settings.URL_PREDICT


@router.post("", response_model=SummaryView)
def create_summary(
    summary_in: SummaryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    summary = Summary(
        id=uuid.uuid4(),
        user_id=current_user.id,
        original_text=summary_in.original_text,
        summary_text=summary_in.summary_text,
        created_at=dt.datetime.now(dt.timezone.utc),
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary

@router.get("", response_model=list[SummaryView])
def get_summaries(
    skip: int = 0,
    limit: int = Query(10, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    summaries = (
        db.query(Summary)
        .filter(Summary.user_id == current_user.id)
        .order_by(desc(Summary.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
    return summaries

@router.get("/{summary_id}", response_model=SummaryView)
def get_summary(
    summary_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    summary = db.query(Summary).filter(Summary.id == summary_id, Summary.user_id == current_user.id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    return summary

@router.post("/predict", response_model=SummaryView, status_code=201)
def predict_and_save(
    body: PredictIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Validate cơ bản (Pydantic đã check min/max length rồi)
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Thiếu nội dung cần tóm tắt.")

    # Gọi mô hình nội bộ
    try:
        summary_text, _latency_ms = summarize(text)
    except FileNotFoundError as e:
        # Model chưa sẵn sàng/đường dẫn sai
        raise HTTPException(status_code=503, detail=f"Model chưa sẵn sàng: {e}")
    except Exception as e:
        # Lỗi suy luận
        raise HTTPException(status_code=502, detail=f"Inference error: {e}")

    if not isinstance(summary_text, str) or not summary_text.strip():
        raise HTTPException(status_code=502, detail="Kết quả tóm tắt không hợp lệ.")

    # Lưu DB
    row = Summary(
        id=uuid.uuid4(),
        user_id=str(current_user.id),                
        original_text=text,
        summary_text=summary_text,
        created_at=dt.datetime.now(dt.timezone.utc),
    )
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB insert error: {e}")

    return row