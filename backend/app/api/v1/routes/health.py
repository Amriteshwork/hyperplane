"""/api/v1/health — reachability of the API's dependencies."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.core.deps import get_health_service
from app.schemas.health import HealthStatus, OverallState
from app.services.health_service import HealthService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    operation_id="getHealth",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthStatus}},
)
async def get_health(
    response: Response, service: Annotated[HealthService, Depends(get_health_service)]
) -> HealthStatus:
    """Status of db, redis and storage. 503 if any is down."""
    result = await service.check()
    if result.status is not OverallState.OK:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
