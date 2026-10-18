from pathlib import Path

import pytest
import yaml

from app.samples.catalog import SampleCatalog, UnknownSampleError
from tests.conftest import DATASETS_DIR

META = {
    "id": "toy",
    "name": "Toy",
    "description": "d",
    "task": "regression",
    "target": "y",
    "rows": 1,
    "columns": 2,
    "vendored": True,
    "licence": "CC0 1.0",
    "source": "s",
    "url": None,
    "citation": None,
}


def test_loads_repo_samples() -> None:
    catalog = SampleCatalog.load(DATASETS_DIR)
    ids = {s.meta.id for s in catalog.list()}
    assert {"iris", "penguins", "synthetic-linear", "diabetes", "california-housing"} <= ids
    assert catalog.get("iris").status == "available"


def test_status_reflects_missing_data_file(tmp_path: Path) -> None:
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "meta.yaml").write_text(yaml.safe_dump(META))
    catalog = SampleCatalog.load(tmp_path)
    assert catalog.get("toy").status == "not-fetched"
    with pytest.raises(UnknownSampleError):
        catalog.get("nope")


def test_rejects_id_folder_mismatch(tmp_path: Path) -> None:
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "meta.yaml").write_text(yaml.safe_dump(META))
    with pytest.raises(ValueError, match="folder"):
        SampleCatalog.load(tmp_path)


def test_missing_root_is_empty(tmp_path: Path) -> None:
    assert SampleCatalog.load(tmp_path / "absent").list() == []
