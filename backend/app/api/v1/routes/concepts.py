"""/api/v1/concepts — endpoints arrive with their M0 milestone (ARCHITECTURE §6)."""

from fastapi import APIRouter

router = APIRouter(prefix="/concepts", tags=["concepts"])
