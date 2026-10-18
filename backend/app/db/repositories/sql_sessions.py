from datetime import datetime
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SessionRow
from app.domain.sessions import Session


class SqlSessionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get(self, session_id: UUID) -> Session | None:
        row = await self._db.get(SessionRow, session_id)
        if row is None:
            return None
        return Session(id=row.id, created_at=row.created_at, last_seen=row.last_seen)

    async def add(self, session: Session) -> None:
        self._db.add(
            SessionRow(id=session.id, created_at=session.created_at, last_seen=session.last_seen)
        )
        await self._db.commit()

    async def touch(self, session_id: UUID, when: datetime) -> None:
        await self._db.execute(
            update(SessionRow).where(SessionRow.id == session_id).values(last_seen=when)
        )
        await self._db.commit()
