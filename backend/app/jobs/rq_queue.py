"""`JobQueue` backed by RQ."""

from redis import Redis
from rq import Queue

from app.jobs.base import JsonValue


class RqJobQueue:
    """Jobs are referenced by dotted path, so the API never imports worker code."""

    def __init__(self, connection: Redis, *, name: str, timeout_seconds: int) -> None:
        self._queue = Queue(name, connection=connection)
        self._timeout = timeout_seconds

    def enqueue(self, job: str, /, **kwargs: JsonValue) -> str:
        return self._queue.enqueue(job, kwargs=kwargs, job_timeout=self._timeout).id
