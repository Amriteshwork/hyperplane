"""/api/v1/experiments — endpoints arrive with their M0 milestone (ARCHITECTURE §6)."""

from fastapi import APIRouter

router = APIRouter(prefix="/experiments", tags=["experiments"])
