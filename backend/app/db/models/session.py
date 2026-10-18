from uuid import UUID

from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models._types import Timestamp


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    created_at: Mapped[Timestamp]
    last_seen: Mapped[Timestamp]
