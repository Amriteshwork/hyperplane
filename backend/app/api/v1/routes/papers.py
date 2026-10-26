"""/api/v1/papers — section-by-section walkthroughs of papers and blog posts (ADR 0022)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.deps import get_content_service
from app.schemas.common import Page, problem_responses
from app.schemas.content import PaperOut, PaperSummaryOut
from app.services.content_service import ContentService

router = APIRouter(prefix="/papers", tags=["learn"])


@router.get("", operation_id="listPapers")
async def list_papers(
    service: Annotated[ContentService, Depends(get_content_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[PaperSummaryOut]:
    papers = service.list_papers()
    return Page(
        items=list(papers[offset : offset + limit]), total=len(papers), limit=limit, offset=offset
    )


@router.get("/{paper_id}", operation_id="getPaper", responses=problem_responses(404))
async def get_paper(
    paper_id: str, service: Annotated[ContentService, Depends(get_content_service)]
) -> PaperOut:
    return service.get_paper(paper_id)
