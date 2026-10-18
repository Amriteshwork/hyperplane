"""Shared response shapes."""

from pydantic import BaseModel

from app.schemas.problem import PROBLEM_MEDIA_TYPE, Problem


class Page[T](BaseModel):
    """A page of a list endpoint (ARCHITECTURE §6: every list is paginated)."""

    items: list[T]
    total: int
    limit: int
    offset: int


def problem_responses(*statuses: int) -> dict[int | str, dict[str, object]]:
    """OpenAPI `responses=` entries documenting problem+json errors."""
    return {status: {"model": Problem, "content": {PROBLEM_MEDIA_TYPE: {}}} for status in statuses}
