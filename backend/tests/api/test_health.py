from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.deps import get_health_service
from app.services.health_service import HealthService
from tests.fakes.health import StaticProbe


def _use_probes(app: FastAPI, *probes: StaticProbe) -> None:
    app.dependency_overrides[get_health_service] = lambda: HealthService(
        list(probes), timeout_seconds=1.0
    )


def test_all_services_up(app: FastAPI, client: TestClient) -> None:
    _use_probes(app, StaticProbe("db"), StaticProbe("redis"), StaticProbe("storage"))

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert {name: s["status"] for name, s in body["services"].items()} == {
        "db": "up",
        "redis": "up",
        "storage": "up",
    }
    assert response.headers["X-Request-ID"]


def test_degraded_when_a_service_is_down(app: FastAPI, client: TestClient) -> None:
    _use_probes(app, StaticProbe("db"), StaticProbe("redis", ConnectionError("refused")))

    response = client.get("/api/v1/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"]["redis"] == {
        "status": "down",
        "latency_ms": body["services"]["redis"]["latency_ms"],
        "error": "ConnectionError",
    }


def test_real_wiring_reports_each_dependency(client: TestClient) -> None:
    # Default composition root: local storage works, db and redis point at a closed port.
    response = client.get("/api/v1/health")

    assert response.status_code == 503
    services = response.json()["services"]
    assert set(services) == {"db", "redis", "storage"}
    assert services["storage"]["status"] == "up"
    assert services["db"]["status"] == "down"
    assert services["redis"]["status"] == "down"


def test_health_is_in_openapi(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/api/v1/health"]["get"]
    assert operation["operationId"] == "getHealth"
    assert {"200", "503"} <= set(operation["responses"])
