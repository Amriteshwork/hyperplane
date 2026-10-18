from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models._types import Json, Timestamp


class ExperimentRow(Base):
    """A training setup on one dataset; repositories arrive with the training pipeline."""

    __tablename__ = "experiments"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    target: Mapped[str] = mapped_column(String(255))
    features: Mapped[list[str]] = mapped_column(Json)
    task_type: Mapped[str] = mapped_column(String(16))
    preprocessing: Mapped[dict[str, Any]] = mapped_column(Json)
    split: Mapped[dict[str, Any]] = mapped_column(Json)
    seed: Mapped[int]
    created_at: Mapped[Timestamp]
