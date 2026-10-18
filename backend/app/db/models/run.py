from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models._types import Json, Timestamp


class RunRow(Base):
    """One training of one algorithm for an experiment."""

    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    experiment_id: Mapped[UUID] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    algorithm_id: Mapped[str] = mapped_column(String(64), index=True)
    algorithm_version: Mapped[str] = mapped_column(String(32))
    params: Mapped[dict[str, Any]] = mapped_column(Json)
    status: Mapped[str] = mapped_column(String(16), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metrics: Mapped[dict[str, Any] | None] = mapped_column(Json)
    plots: Mapped[dict[str, Any] | None] = mapped_column(Json)
    artifact_key: Mapped[str | None] = mapped_column(String(512))
    error: Mapped[str | None] = mapped_column(Text)
    env: Mapped[dict[str, Any] | None] = mapped_column(Json)
    created_at: Mapped[Timestamp]
