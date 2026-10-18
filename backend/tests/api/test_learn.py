from fastapi.testclient import TestClient


def test_families_list_available_and_upcoming_algorithms(client: TestClient) -> None:
    response = client.get("/api/v1/families")
    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["id"] == "linear-models"  # ordered by `order`
    algorithms = {a["id"]: a for a in body["items"][0]["algorithms"]}
    assert algorithms["linear_regression"]["available"] is True
    assert algorithms["linear_regression"]["task_types"] == ["regression"]
    assert algorithms["ridge"] == {
        "id": "ridge",
        "name": "Ridge",
        "available": False,
        "summary": None,
        "task_types": [],
        "difficulty": None,
    }


def test_algorithm_page(client: TestClient) -> None:
    body = client.get("/api/v1/algorithms/linear_regression").json()
    assert body["name"] == "Linear Regression"
    assert body["family"]["id"] == "linear-models"
    assert "## Intuition" in body["overview"]
    assert {"id": "normal-equation", "title": "Normal equation"} in body["concepts"]
    assert body["resources"] == []
    assert body["params_schema"]["properties"]["fit_intercept"]["level"] == "beginner"
    assert [v["id"] for v in body["code_variants"]] == ["train", "scratch"]


def test_example_code_has_resolved_annotations(client: TestClient) -> None:
    body = client.get("/api/v1/algorithms/linear_regression/code/train").json()
    lines = body["code"].splitlines()
    fit = next(a for a in body["annotations"] if a["anchor"] == "fit")

    assert body["source"] == "example"
    assert body["filename"] == "train.py"
    assert "@a:" not in body["code"]
    assert lines[fit["start_line"] - 1] == "pipeline.fit(X_train, y_train)"
    assert {"id": "mean-squared-error", "title": "Mean squared error (MSE)"} in fit["concepts"]


def test_concept_page_links_algorithms_and_prerequisites(client: TestClient) -> None:
    body = client.get("/api/v1/concepts/normal-equation").json()
    assert body["title"] == "Normal equation"
    assert body["prerequisites"] == [
        {"id": "mean-squared-error", "title": "Mean squared error (MSE)"}
    ]
    assert body["algorithms"] == [{"id": "linear_regression", "name": "Linear Regression"}]


def test_unknown_ids_are_404_problems(client: TestClient) -> None:
    for path in (
        "/api/v1/algorithms/ridge",
        "/api/v1/algorithms/linear_regression/code/nope",
        "/api/v1/concepts/nope",
    ):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.headers["content-type"].startswith("application/problem+json")
