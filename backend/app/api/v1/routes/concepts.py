"""/api/v1/concepts — concept pages."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.deps import get_content_service
from app.schemas.common import problem_responses
from app.schemas.content import ConceptOut
from app.services.content_service import ContentService

router = APIRouter(prefix="/concepts", tags=["learn"])


@router.get("/{concept_id}", operation_id="getConcept", responses=problem_responses(404))
async def get_concept(
    concept_id: str, service: Annotated[ContentService, Depends(get_content_service)]
) -> ConceptOut:
    return service.get_concept(concept_id)
