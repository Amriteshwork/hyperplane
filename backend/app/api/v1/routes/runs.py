"""/api/v1/runs — a run's status, results and live progress."""

from collections.abc import AsyncIterable, AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse

from app.core.deps import SessionId, get_experiment_service
from app.schemas.common import problem_responses
from app.schemas.experiments import RunEventOut, RunOut
from app.services.experiment_service import ExperimentService, RunEvents

router = APIRouter(prefix="/runs", tags=["runs"])
Service = Annotated[ExperimentService, Depends(get_experiment_service)]


async def open_run_events(
    run_id: UUID, session_id: SessionId, service: Service
) -> AsyncIterator[RunEvents]:
    """Checks access before the stream starts; releases the subscription when it ends."""
    events = await service.open_events(session_id, run_id)
    try:
        yield events
    finally:
        await events.aclose()


@router.get("/{run_id}", operation_id="getRun", responses=problem_responses(404))
async def get_run(run_id: UUID, session_id: SessionId, service: Service) -> RunOut:
    return RunOut.from_domain(await service.get_run(session_id, run_id))


@router.get(
    "/{run_id}/events",
    operation_id="streamRunEvents",
    response_class=EventSourceResponse,
    response_model_exclude_none=True,
    responses=problem_responses(404),
)
async def stream_run_events(
    events: Annotated[RunEvents, Depends(open_run_events)],
) -> AsyncIterable[RunEventOut]:
    """Server-sent events (`status`, `progress`, then `done` or `error`) until the run ends."""
    async for event in events:
        yield RunEventOut.model_validate(event)
