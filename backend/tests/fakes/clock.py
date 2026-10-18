from datetime import UTC, datetime, timedelta


class FixedClock:
    """A clock tests move by hand."""

    def __init__(self, now: datetime | None = None) -> None:
        self.current = now or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta
