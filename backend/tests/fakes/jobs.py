"""`FakeJobQueue`: records enqueued jobs; `drain()` runs them with registered handlers."""

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from app.jobs.base import JsonValue


@dataclass(frozen=True, slots=True)
class EnqueuedJob:
    id: str
    name: str
    kwargs: dict[str, JsonValue]


class QueueUnavailableError(ConnectionError):
    pass


class FakeJobQueue:
    def __init__(self) -> None:
        self.jobs: list[EnqueuedJob] = []
        self.handlers: dict[str, Callable[..., None]] = {}
        self.fail = False

    def enqueue(self, job: str, /, **kwargs: JsonValue) -> str:
        if self.fail:
            raise QueueUnavailableError("queue down")
        enqueued = EnqueuedJob(id=str(uuid4()), name=job, kwargs=kwargs)
        self.jobs.append(enqueued)
        return enqueued.id

    def drain(self) -> None:
        """Run and remove every pending job, in order (call from the test thread)."""
        while self.jobs:
            job = self.jobs.pop(0)
            self.handlers[job.name](**job.kwargs)
