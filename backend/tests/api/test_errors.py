"""Every error leaves the API as problem+json carrying the request id."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import NotFoundError


@pytest.fixture
def app(app: FastAPI) -> FastAPI:
    @app.get("/test/items/{item_id}")
    async def get_item(item_id: int) -> dict[str, int]:
        if item_id == 0:
            raise NotFoundError("Item 0 does not exist.")
        if item_id == 500:
            raise RuntimeError("boom")
        return {"id": item_id}

    return app


def test_unknown_route_is_problem_json(client: TestClient) -> None:
    response = client.get("/nope")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["title"] == "Not Found"
    assert body["status"] == 404
    assert body["instance"] == "/nope"
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_app_error_maps_to_its_status(client: TestClient) -> None:
    body = client.get("/test/items/0").json()
    assert body["status"] == 404
    assert body["detail"] == "Item 0 does not exist."


def test_validation_error_lists_fields(client: TestClient) -> None:
    response = client.get("/test/items/abc")

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["errors"][0]["loc"] == ["path", "item_id"]


def test_unhandled_error_hides_internals(client: TestClient) -> None:
    response = client.get("/test/items/500")

    assert response.status_code == 500
    body = response.json()
    assert body["title"] == "Internal Server Error"
    assert "boom" not in response.text
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_incoming_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/test/items/1", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_unsafe_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/test/items/1", headers={"X-Request-ID": "bad id\r\n"})
    assert response.headers["X-Request-ID"] != "bad id"
    assert len(response.headers["X-Request-ID"]) == 32
