from datetime import timedelta
from uuid import uuid4

import pytest

from app.services.session_service import SessionService
from tests.fakes.clock import FixedClock
from tests.fakes.repositories import InMemorySessionRepository

pytestmark = pytest.mark.anyio


@pytest.fixture
def service(session_repo: InMemorySessionRepository, clock: FixedClock) -> SessionService:
    return SessionService(session_repo, clock)


@pytest.mark.parametrize("raw", [None, "", "not-a-uuid", str(uuid4())])
async def test_creates_session_when_missing_or_unknown(
    service: SessionService, session_repo: InMemorySessionRepository, raw: str | None
) -> None:
    session_id = await service.resolve(raw)
    assert session_id in session_repo.rows
    assert str(session_id) != raw


async def test_reuses_known_session_and_touches_it_lazily(
    service: SessionService, session_repo: InMemorySessionRepository, clock: FixedClock
) -> None:
    session_id = await service.resolve(None)
    created = session_repo.rows[session_id].last_seen

    clock.advance(timedelta(minutes=1))
    assert await service.resolve(str(session_id)) == session_id
    assert session_repo.rows[session_id].last_seen == created  # within touch interval

    clock.advance(timedelta(minutes=10))
    await service.resolve(str(session_id))
    assert session_repo.rows[session_id].last_seen == clock.now()
