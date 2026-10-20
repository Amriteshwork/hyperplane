import json

import pytest
from fastapi.testclient import TestClient

from app.domain.experiments import RunStatus
from app.storage.base import BlobNotFoundError
from tests.fakes.events import InMemoryEventBus
from tests.fakes.jobs import FakeJobQueue
from tests.fakes.repositories import InMemoryRunRepository
from tests.fakes.storage import InMemoryStorage

CSV = b"x1,x2,colour,y\n" + b"".join(
    f"{i * 7 % 40},{i * 2 % 7},{['red', 'blue'][i % 2]},{i * 1.5 + (i % 3)}\n".encode()
    for i in range(40)
)
TRAIN = {"target": "y", "features": ["x1", "x2", "colour"], "algorithm_id": "linear_regression"}


def _dataset(client: TestClient, job_queue: FakeJobQueue) -> str:
    response = client.post("/api/v1/datasets", files={"file": ("d.csv", CSV)})
    job_queue.drain()
    dataset_id: str = response.json()["id"]
    return dataset_id


def _events(client: TestClient, run_id: str) -> list[dict[str, object]]:
    response = client.get(f"/api/v1/runs/{run_id}/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_train_a_run_end_to_end(
    client: TestClient,
    job_queue: FakeJobQueue,
    storage: InMemoryStorage,
    event_bus: InMemoryEventBus,
) -> None:
    dataset_id = _dataset(client, job_queue)
    created = client.post("/api/v1/experiments", json={"dataset_id": dataset_id, **TRAIN})
    assert created.status_code == 201, created.text
    run = created.json()
    assert run["status"] == "queued"
    assert run["params"] == {"fit_intercept": True, "positive": False}
    assert run["experiment"]["numeric_features"] == ["x1", "x2"]
    assert run["experiment"]["categorical_features"] == ["colour"]
    assert (run["experiment"]["task_type"], run["experiment"]["test_size"]) == ("regression", 0.2)
    assert [j.name for j in job_queue.jobs] == ["app.workers.jobs.train_run"]

    job_queue.drain()
    done = client.get(f"/api/v1/runs/{run['id']}").json()
    assert done["status"] == "succeeded" and done["error"] is None
    report = done["report"]
    assert (report["n_train"], report["n_test"]) == (32, 8)
    assert [m["id"] for m in report["evaluation"]["metrics"]] == ["r2", "rmse", "mae"]
    assert [p["id"] for p in report["evaluation"]["plots"]] == ["predicted-vs-actual", "residuals"]
    assert len(report["coefficients"]) == 4  # x1, x2, colour_blue, colour_red
    assert storage.read(f"runs/{run['id']}/model.joblib")

    published = event_bus.published[f"runs:{run['id']}:events"]
    assert [e["type"] for e in published] == ["status", "progress", "progress", "progress", "done"]
    assert _events(client, run["id"]) == [{"type": "done", "status": "succeeded"}]


def test_train_a_classifier_end_to_end(client: TestClient, job_queue: FakeJobQueue) -> None:
    dataset_id = _dataset(client, job_queue)
    body = {"target": "colour", "features": ["x1", "x2"], "algorithm_id": "logistic_regression"}
    created = client.post("/api/v1/experiments", json={"dataset_id": dataset_id, **body})
    assert created.status_code == 201, created.text
    assert created.json()["experiment"]["task_type"] == "binary"

    job_queue.drain()
    done = client.get(f"/api/v1/runs/{created.json()['id']}").json()
    assert done["status"] == "succeeded", done["error"]
    evaluation = done["report"]["evaluation"]
    metrics = [m["id"] for m in evaluation["metrics"]]
    assert metrics[:5] == ["accuracy", "precision", "recall", "f1", "log_loss"]
    assert evaluation["plots"][0]["kind"] == "heatmap"
    assert {c["target_class"] for c in done["report"]["coefficients"]} == {"red"}  # classes_[1]


@pytest.mark.parametrize(
    ("changes", "detail"),
    [
        ({"target": "nope"}, "no column 'nope'"),
        ({"target": "colour", "features": ["x1"]}, "can't be used for binary tasks"),
        ({"features": ["x1", "y"]}, "target can't also be a feature"),
        ({"features": ["x1", "zzz"]}, "Unknown feature columns: zzz"),
        ({"features": ["x1", "x1"]}, "only once"),
        ({"algorithm_id": "nope"}, "Unknown algorithm 'nope'"),
        ({"params": {"fit_intercept": "maybe"}}, "Invalid hyperparameters: fit_intercept"),
        ({"params": {"alpha": 1}}, "Invalid hyperparameters: alpha"),
    ],
)
def test_invalid_setups_are_rejected(
    client: TestClient, job_queue: FakeJobQueue, changes: dict[str, object], detail: str
) -> None:
    dataset_id = _dataset(client, job_queue)
    response = client.post(
        "/api/v1/experiments", json={"dataset_id": dataset_id, **TRAIN, **changes}
    )
    assert response.status_code == 422, response.text
    assert detail in response.json()["detail"]
    assert job_queue.jobs == []


def test_dataset_must_be_profiled_and_owned(client: TestClient, job_queue: FakeJobQueue) -> None:
    dataset_id = client.post("/api/v1/datasets", files={"file": ("d.csv", CSV)}).json()["id"]
    body = {"dataset_id": dataset_id, **TRAIN}
    assert client.post("/api/v1/experiments", json=body).status_code == 409  # still profiling

    job_queue.drain()
    client.cookies.clear()
    assert client.post("/api/v1/experiments", json=body).status_code == 404


def test_runs_are_private_to_their_session(client: TestClient, job_queue: FakeJobQueue) -> None:
    dataset_id = _dataset(client, job_queue)
    run_id = client.post("/api/v1/experiments", json={"dataset_id": dataset_id, **TRAIN}).json()[
        "id"
    ]
    client.cookies.clear()
    assert client.get(f"/api/v1/runs/{run_id}").status_code == 404
    assert client.get(f"/api/v1/runs/{run_id}/events").status_code == 404


def test_queue_outage_fails_the_run(
    client: TestClient, job_queue: FakeJobQueue, run_repo: InMemoryRunRepository
) -> None:
    dataset_id = _dataset(client, job_queue)
    job_queue.fail = True
    response = client.post("/api/v1/experiments", json={"dataset_id": dataset_id, **TRAIN})

    assert response.status_code == 503
    [run] = run_repo.runs.values()
    assert run.status is RunStatus.FAILED


def test_failed_training_is_reported(
    client: TestClient, job_queue: FakeJobQueue, storage: InMemoryStorage
) -> None:
    dataset_id = _dataset(client, job_queue)
    run_id = client.post("/api/v1/experiments", json={"dataset_id": dataset_id, **TRAIN}).json()[
        "id"
    ]
    storage.delete(f"datasets/{dataset_id}/data.parquet")
    with pytest.raises(BlobNotFoundError):
        job_queue.drain()  # the worker re-raises so RQ records the failure too

    run = client.get(f"/api/v1/runs/{run_id}").json()
    assert run["status"] == "failed"
    assert run["error"].startswith("Training failed: BlobNotFoundError")
    [event] = _events(client, run_id)
    assert event["type"] == "error" and event["message"] == run["error"]


def test_training_options_list_targets_algorithms_and_features(
    client: TestClient, job_queue: FakeJobQueue
) -> None:
    dataset_id = _dataset(client, job_queue)
    options = client.get(f"/api/v1/datasets/{dataset_id}/training-options").json()

    targets = {t["column"]: t for t in options["targets"]}
    assert targets["y"]["task_type"] == "regression"
    assert targets["y"]["algorithms"] == [
        {"id": "linear_regression", "name": "Linear Regression"},
        {"id": "polynomial_regression", "name": "Polynomial Regression"},
        {"id": "ridge", "name": "Ridge"},
        {"id": "lasso", "name": "Lasso"},
        {"id": "elastic_net", "name": "Elastic Net"},
    ]  # in teaching order (the family pages), not by id: the Lab preselects the first
    assert targets["colour"]["algorithms"] == [
        {"id": "logistic_regression", "name": "Logistic Regression"}
    ]
    assert {f["column"]: f["role"] for f in options["features"]} == {
        "x1": "numeric",
        "x2": "numeric",
        "colour": "categorical",
        "y": "numeric",
    }
