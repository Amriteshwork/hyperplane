"""/api/v1/samples — built-in datasets."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.core.deps import SessionId, get_dataset_service, get_sample_catalog
from app.core.errors import NotFoundError
from app.samples.catalog import SampleCatalog, SampleDataset, UnknownSampleError
from app.schemas.common import Page, problem_responses
from app.schemas.datasets import DatasetOut
from app.schemas.samples import SampleOut
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/samples", tags=["samples"])
Catalog = Annotated[SampleCatalog, Depends(get_sample_catalog)]


@router.get("", operation_id="listSamples")
async def list_samples(
    catalog: Catalog,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[SampleOut]:
    samples = catalog.list()
    items = [SampleOut.from_domain(s) for s in samples[offset : offset + limit]]
    return Page(items=items, total=len(samples), limit=limit, offset=offset)


@router.post(
    "/{sample_id}/datasets",
    status_code=status.HTTP_201_CREATED,
    operation_id="createDatasetFromSample",
    responses=problem_responses(404, 409),
)
async def create_dataset_from_sample(
    sample_id: str,
    session_id: SessionId,
    catalog: Catalog,
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetOut:
    """Copy a sample into the session as a new dataset (then profiled like an upload)."""
    return DatasetOut.from_domain(
        await service.create_from_sample(session_id, _find(catalog, sample_id))
    )


def _find(catalog: SampleCatalog, sample_id: str) -> SampleDataset:
    try:
        return catalog.get(sample_id)
    except UnknownSampleError:
        raise NotFoundError(f"Unknown sample {sample_id!r}.") from None
