from fastapi.testclient import TestClient

from tests.fakes.jobs import FakeJobQueue


def test_list_samples(client: TestClient) -> None:
    body = client.get("/api/v1/samples").json()
    by_id = {s["id"]: s for s in body["items"]}
    assert body["total"] == len(body["items"]) >= 7
    assert by_id["iris"]["task"] == "multiclass" and by_id["iris"]["status"] == "available"
    assert by_id["synthetic-linear"]["target"] == "y"


def test_list_samples_paginates(client: TestClient) -> None:
    page = client.get("/api/v1/samples", params={"limit": 2, "offset": 1}).json()
    assert len(page["items"]) == 2 and page["offset"] == 1


def test_create_dataset_from_sample(client: TestClient, job_queue: FakeJobQueue) -> None:
    response = client.post("/api/v1/samples/iris/datasets")
    assert response.status_code == 201
    created = response.json()
    assert created["source"] == "sample" and created["sample_id"] == "iris"
    assert (created["n_rows"], created["n_columns"]) == (150, 5)

    job_queue.drain()
    profile = client.get(f"/api/v1/datasets/{created['id']}/profile").json()
    assert profile["target_candidates"][0]["column"] == "species"


def test_unknown_sample(client: TestClient) -> None:
    assert client.post("/api/v1/samples/nope/datasets").status_code == 404
