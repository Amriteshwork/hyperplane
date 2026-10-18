"""Anonymous sessions: resolve a client-supplied id or start a new session."""

from datetime import timedelta
from uuid import UUID, uuid4

from app.core.clock import Clock
from app.db.repositories.base import SessionRepository
from app.domain.sessions import Session

# Avoid a DB write per request: last_seen only moves when it is older than this.
_TOUCH_INTERVAL = timedelta(minutes=5)


class SessionService:
    def __init__(self, sessions: SessionRepository, clock: Clock) -> None:
        self._sessions = sessions
        self._clock = clock

    async def resolve(self, raw_id: str | None) -> UUID:
        """Return the existing session for `raw_id`, or create one if it is missing/unknown."""
        now = self._clock.now()
        session_id = _parse(raw_id)
        if session_id is not None:
            session = await self._sessions.get(session_id)
            if session is not None:
                if now - session.last_seen > _TOUCH_INTERVAL:
                    await self._sessions.touch(session.id, now)
                return session.id
        session = Session(id=uuid4(), created_at=now, last_seen=now)
        await self._sessions.add(session)
        return session.id


def _parse(raw_id: str | None) -> UUID | None:
    try:
        return UUID(raw_id) if raw_id else None
    except ValueError:
        return None
