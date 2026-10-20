"""/api/v1/experiments — create a training run."""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.deps import SessionId, get_experiment_service
from app.schemas.common import problem_responses
from app.schemas.experiments import ExperimentIn, RunOut
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["experiments"])
Service = Annotated[ExperimentService, Depends(get_experiment_service)]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    operation_id="createExperiment",
    responses=problem_responses(404, 409, 422, 503),
)
async def create_experiment(body: ExperimentIn, session_id: SessionId, service: Service) -> RunOut:
    """Queue a training run; follow it with `GET /runs/{id}/events`."""
    return RunOut.from_domain(await service.create(session_id, body))
