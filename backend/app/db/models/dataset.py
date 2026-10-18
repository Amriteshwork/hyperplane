from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models._types import Json, Timestamp


class DatasetRow(Base):
    __tablename__ = "datasets"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(16))
    sample_id: Mapped[str | None] = mapped_column(String(64))
    format: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    raw_key: Mapped[str] = mapped_column(String(512))
    parquet_key: Mapped[str] = mapped_column(String(512))
    n_rows: Mapped[int]
    n_columns: Mapped[int]
    status: Mapped[str] = mapped_column(String(16), index=True)
    profile: Mapped[dict[str, Any] | None] = mapped_column(Json)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[Timestamp]
    expires_at: Mapped[Timestamp] = mapped_column(index=True)
