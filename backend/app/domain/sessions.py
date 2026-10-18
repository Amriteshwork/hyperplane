"""Anonymous workspace sessions (ADR 0007)."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class Session:
    id: UUID
    created_at: datetime
    last_seen: datetime
