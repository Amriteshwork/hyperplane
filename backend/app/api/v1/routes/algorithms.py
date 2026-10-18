"""/api/v1/algorithms — algorithm pages and their explained example code."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.deps import get_content_service
from app.schemas.common import problem_responses
from app.schemas.content import AlgorithmOut, CodeOut
from app.services.content_service import ContentService

router = APIRouter(prefix="/algorithms", tags=["learn"])
Service = Annotated[ContentService, Depends(get_content_service)]


@router.get("/{algorithm_id}", operation_id="getAlgorithm", responses=problem_responses(404))
async def get_algorithm(algorithm_id: str, service: Service) -> AlgorithmOut:
    return service.get_algorithm(algorithm_id)


@router.get(
    "/{algorithm_id}/code/{variant}",
    operation_id="getAlgorithmCode",
    responses=problem_responses(404),
)
async def get_algorithm_code(algorithm_id: str, variant: str, service: Service) -> CodeOut:
    """A code variant (e.g. `train`, `scratch`) rendered with example settings."""
    return service.get_example_code(algorithm_id, variant)
