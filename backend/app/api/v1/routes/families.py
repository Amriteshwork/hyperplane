"""/api/v1/families — algorithm families with their (available and upcoming) algorithms."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.deps import get_content_service
from app.schemas.common import Page
from app.schemas.content import FamilyOut
from app.services.content_service import ContentService

router = APIRouter(prefix="/families", tags=["learn"])


@router.get("", operation_id="listFamilies")
async def list_families(
    service: Annotated[ContentService, Depends(get_content_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[FamilyOut]:
    families = service.list_families()
    return Page(
        items=list(families[offset : offset + limit]),
        total=len(families),
        limit=limit,
        offset=offset,
    )
