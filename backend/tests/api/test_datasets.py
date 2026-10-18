import io
from datetime import timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.domain.datasets import DatasetStatus
from tests.fakes.clock import FixedClock
from tests.fakes.jobs import FakeJobQueue
from tests.fakes.repositories import InMemoryDatasetRepository

CSV = b"x1,x2,colour,y\n" + b"".join(
    f"{i},{i * 2 % 7},{['red', 'blue'][i % 2]},{i * 1.5 + 0.25}\n".encode() for i in range(30)
)


def _upload(client: TestClient, data: bytes, name: str = "data.csv") -> dict[str, object]:
    response = client.post("/api/v1/datasets", files={"file": (name, data)})
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def test_upload_profile_preview(client: TestClient, job_queue: FakeJobQueue) -> None:
    created = _upload(client, CSV)
    assert created["status"] == "profiling" and created["format"] == "csv"
    assert (created["n_rows"], created["n_columns"]) == (30, 4)
    assert [j.name for j in job_queue.jobs] == ["app.workers.jobs.profile_dataset"]

    url = f"/api/v1/datasets/{created['id']}"
    assert client.get(f"{url}/profile").status_code == 409  # not profiled yet

    job_queue.drain()
    assert client.get(url).json()["status"] == "ready"
    profile = client.get(f"{url}/profile").json()
    assert profile["target_candidates"][0] == {
        "column": "y",
        "task_type": "regression",
        "reason": "30 distinct numeric values → regression.",
    }

    preview = client.get(f"{url}/preview", params={"rows": 2}).json()
    assert preview == {
        "columns": ["x1", "x2", "colour", "y"],
        "rows": [[0, 0, "red", 0.25], [1, 2, "blue", 1.75]],
        "total_rows": 30,
    }


def test_session_cookie_and_header(client: TestClient) -> None:
    response = client.post("/api/v1/datasets", files={"file": ("d.csv", CSV)})
    session_id = response.headers["X-Session-ID"]
    assert client.cookies.get("hp_session") == session_id
    dataset_id = response.json()["id"]

    # A different session can't see the dataset.
    client.cookies.clear()
    assert client.get(f"/api/v1/datasets/{dataset_id}").status_code == 404
    # The header works as an alternative to the cookie.
    client.cookies.clear()
    own = client.get(f"/api/v1/datasets/{dataset_id}", headers={"X-Session-ID": session_id})
    assert own.status_code == 200


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(index=False),
        None,  # xlsx, built below
    ],
    ids=["parquet", "xlsx"],
)
def test_upload_other_formats(client: TestClient, data: bytes | None) -> None:
    if data is None:
        buffer = io.BytesIO()
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buffer, index=False)
        data = buffer.getvalue()
    created = _upload(client, data, name="whatever.bin")  # format comes from bytes, not name
    assert (created["n_rows"], created["n_columns"]) == (2, 2)


@pytest.mark.parametrize(
    ("data", "status"),
    [
        (b"", 415),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, 415),
        (b"a,b\n", 422),
    ],
    ids=["empty", "png", "header-only"],
)
def test_rejects_bad_files(client: TestClient, data: bytes, status: int) -> None:
    response = client.post("/api/v1/datasets", files={"file": ("x.csv", data)})
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize("settings_overrides", [{"max_upload_mb": 1}])
def test_rejects_oversized_upload(client: TestClient, job_queue: FakeJobQueue) -> None:
    data = b"x\n" + b"1234567\n" * (2**20 // 8 + 10)
    response = client.post("/api/v1/datasets", files={"file": ("big.csv", data)})
    assert response.status_code == 413
    assert job_queue.jobs == []


@pytest.mark.parametrize("settings_overrides", [{"max_columns": 3, "max_rows": 10}])
@pytest.mark.parametrize(
    ("data", "message"),
    [(b"a,b,c,d\n1,2,3,4\n", "columns"), (b"a\n" + b"1\n" * 11, "rows")],
)
def test_rejects_tables_over_limits(client: TestClient, data: bytes, message: str) -> None:
    response = client.post("/api/v1/datasets", files={"file": ("x.csv", data)})
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_queue_failure_marks_dataset_failed(
    client: TestClient, job_queue: FakeJobQueue, dataset_repo: InMemoryDatasetRepository
) -> None:
    job_queue.fail = True
    response = client.post("/api/v1/datasets", files={"file": ("d.csv", CSV)})
    assert response.status_code == 503
    (dataset,) = dataset_repo.rows.values()
    assert dataset.status is DatasetStatus.FAILED


def test_profiling_failure_is_reported(
    client: TestClient, job_queue: FakeJobQueue, storage: object
) -> None:
    created = _upload(client, CSV)
    storage.blobs.pop(f"datasets/{created['id']}/data.parquet")  # type: ignore[attr-defined]
    with pytest.raises(KeyError):
        job_queue.drain()
    url = f"/api/v1/datasets/{created['id']}"
    assert client.get(url).json()["status"] == "failed"
    assert client.get(f"{url}/profile").status_code == 409


def test_expired_dataset_is_gone(client: TestClient, clock: FixedClock) -> None:
    created = _upload(client, CSV)
    clock.advance(timedelta(days=8))
    assert client.get(f"/api/v1/datasets/{created['id']}").status_code == 404


@pytest.mark.parametrize("rows", [0, 501])
def test_preview_row_bounds(client: TestClient, rows: int) -> None:
    created = _upload(client, CSV)
    response = client.get(f"/api/v1/datasets/{created['id']}/preview", params={"rows": rows})
    assert response.status_code == 422


def test_unknown_and_malformed_ids(client: TestClient) -> None:
    assert client.get("/api/v1/datasets/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/api/v1/datasets/nope").status_code == 422
