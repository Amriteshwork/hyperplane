"""RFC 9457 (formerly 7807) problem details — the shape of every error response."""

from pydantic import BaseModel, ConfigDict

PROBLEM_MEDIA_TYPE = "application/problem+json"


class Problem(BaseModel):
    """Problem details; extension members (e.g. `errors`) are allowed."""

    model_config = ConfigDict(extra="allow")

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    request_id: str | None = None
