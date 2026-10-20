"""/api/v1/datasets — upload, metadata, profile, preview."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, UploadFile, status

from app.core.deps import SessionId, get_dataset_service, get_experiment_service
from app.ml.profiling import DatasetProfile
from app.schemas.common import problem_responses
from app.schemas.datasets import DatasetOut, DatasetPreviewOut
from app.schemas.experiments import TrainingOptionsOut
from app.services.dataset_service import DatasetService
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/datasets", tags=["datasets"])
Service = Annotated[DatasetService, Depends(get_dataset_service)]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    operation_id="uploadDataset",
    responses=problem_responses(413, 415, 422, 503),
)
async def upload_dataset(file: UploadFile, session_id: SessionId, service: Service) -> DatasetOut:
    """Upload a CSV, Parquet or XLSX file. Profiling runs in the background."""
    # Read one byte past the limit so the service can reject oversized files.
    data = await file.read(service.max_upload_bytes + 1)
    dataset = await service.upload(session_id, file.filename, data)
    return DatasetOut.from_domain(dataset)


@router.get("/{dataset_id}", operation_id="getDataset", responses=problem_responses(404))
async def get_dataset(dataset_id: UUID, session_id: SessionId, service: Service) -> DatasetOut:
    return DatasetOut.from_domain(await service.get(session_id, dataset_id))


@router.get(
    "/{dataset_id}/profile",
    operation_id="getDatasetProfile",
    responses=problem_responses(404, 409),
)
async def get_dataset_profile(
    dataset_id: UUID, session_id: SessionId, service: Service
) -> DatasetProfile:
    """409 while profiling is still running or if it failed."""
    return await service.get_profile(session_id, dataset_id)


@router.get(
    "/{dataset_id}/preview", operation_id="getDatasetPreview", responses=problem_responses(404)
)
async def get_dataset_preview(
    dataset_id: UUID,
    session_id: SessionId,
    service: Service,
    rows: Annotated[int, Query(ge=1, le=500)] = 50,
) -> DatasetPreviewOut:
    return DatasetPreviewOut.from_domain(await service.preview(session_id, dataset_id, rows))


@router.get(
    "/{dataset_id}/training-options",
    operation_id="getTrainingOptions",
    responses=problem_responses(404, 409),
)
async def get_training_options(
    dataset_id: UUID,
    session_id: SessionId,
    experiments: Annotated[ExperimentService, Depends(get_experiment_service)],
) -> TrainingOptionsOut:
    """Targets with the algorithms that can train them, and which columns can be features."""
    return await experiments.training_options(session_id, dataset_id)
