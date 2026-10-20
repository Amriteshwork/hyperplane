"""A run's progress events: published by the worker, streamed to the browser over SSE."""

from enum import StrEnum

from app.domain.experiments import Run, RunStatus
from app.jobs.base import Event
from app.ml.base import Progress


class RunEventType(StrEnum):
    STATUS = "status"
    PROGRESS = "progress"
    DONE = "done"
    ERROR = "error"


FINAL_EVENTS = frozenset({RunEventType.DONE, RunEventType.ERROR})


def status_event(status: RunStatus) -> Event:
    return {"type": RunEventType.STATUS, "status": status}


def progress_event(progress: Progress) -> Event:
    return {"type": RunEventType.PROGRESS, **progress.model_dump()}


def final_event(run: Run) -> Event:
    """The last event of a finished run."""
    if run.status is RunStatus.SUCCEEDED:
        return {"type": RunEventType.DONE, "status": run.status}
    return {"type": RunEventType.ERROR, "status": run.status, "message": run.error}


def is_final(event: Event) -> bool:
    return event.get("type") in FINAL_EVENTS
