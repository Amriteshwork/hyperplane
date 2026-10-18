"""/api/v1/datasets — endpoints arrive with their M0 milestone (ARCHITECTURE §6)."""

from fastapi import APIRouter

router = APIRouter(prefix="/datasets", tags=["datasets"])
