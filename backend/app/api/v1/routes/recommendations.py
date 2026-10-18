"""/api/v1/recommendations — endpoints arrive with their M0 milestone (ARCHITECTURE §6)."""

from fastapi import APIRouter

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
