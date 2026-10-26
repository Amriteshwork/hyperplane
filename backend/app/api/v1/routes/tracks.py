"""/api/v1/tracks — guided routes through the concepts to one goal."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.deps import get_content_service
from app.schemas.common import Page, problem_responses
from app.schemas.content import TrackOut, TrackSummaryOut
from app.services.content_service import ContentService

router = APIRouter(prefix="/tracks", tags=["learn"])


@router.get("", operation_id="listTracks")
async def list_tracks(
    service: Annotated[ContentService, Depends(get_content_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[TrackSummaryOut]:
    tracks = service.list_tracks()
    return Page(
        items=list(tracks[offset : offset + limit]), total=len(tracks), limit=limit, offset=offset
    )


@router.get("/{track_id}", operation_id="getTrack", responses=problem_responses(404))
async def get_track(
    track_id: str, service: Annotated[ContentService, Depends(get_content_service)]
) -> TrackOut:
    return service.get_track(track_id)
