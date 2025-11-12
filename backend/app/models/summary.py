from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import String, DateTime, func, Index
from sqlalchemy.dialects.postgresql import ENUM as PGEnum, UUID
from sqlalchemy import Column, Text
import enum, uuid

class Base(DeclarativeBase):

    pass
def gen_uuid() -> str:
    return str(uuid.uuid4())

class Summary(Base):
    __tablename__ = "summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(36), nullable=False)
    original_text = Column(Text, nullable=True)
    summary_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_summaries_user_created", "user_id", "created_at"),
    )