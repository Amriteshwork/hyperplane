"""/api/v1/runs — endpoints arrive with their M0 milestone (ARCHITECTURE §6)."""

from fastapi import APIRouter

router = APIRouter(prefix="/runs", tags=["runs"])
