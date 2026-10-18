"""/api/v1/algorithms — endpoints arrive with their M0 milestone (ARCHITECTURE §6)."""

from fastapi import APIRouter

router = APIRouter(prefix="/algorithms", tags=["algorithms"])
