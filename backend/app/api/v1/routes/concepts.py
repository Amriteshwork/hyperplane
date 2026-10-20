"""/api/v1/concepts — the concept index (by area) and concept pages."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response

from app.content.models import CodeLibrary
from app.core.deps import get_content_service
from app.schemas.common import Page, problem_responses
from app.schemas.content import ConceptAreaOut, ConceptOut
from app.services.content_service import ContentService

router = APIRouter(prefix="/concepts", tags=["learn"])


@router.get("", operation_id="listConceptAreas")
async def list_concept_areas(
    service: Annotated[ContentService, Depends(get_content_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[ConceptAreaOut]:
    areas = service.list_concept_areas()
    return Page(
        items=list(areas[offset : offset + limit]), total=len(areas), limit=limit, offset=offset
    )


@router.get("/{concept_id}", operation_id="getConcept", responses=problem_responses(404))
async def get_concept(
    concept_id: str, service: Annotated[ContentService, Depends(get_content_service)]
) -> ConceptOut:
    return service.get_concept(concept_id)


@router.get(
    "/{concept_id}/code/{library}/bundle",
    operation_id="getConceptCodeBundle",
    response_class=Response,
    responses={
        200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}},
        **problem_responses(404),
    },
)
async def get_concept_code_bundle(
    concept_id: str,
    library: CodeLibrary,
    service: Annotated[ContentService, Depends(get_content_service)],
) -> Response:
    bundle = service.get_concept_code_bundle(concept_id, library)
    return Response(
        bundle.content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{bundle.filename}"'},
    )
