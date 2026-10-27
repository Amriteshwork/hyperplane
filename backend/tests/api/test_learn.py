import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_families_list_available_and_upcoming_algorithms(client: TestClient) -> None:
    response = client.get("/api/v1/families")
    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["id"] == "linear-models"  # ordered by `order`
    algorithms = {a["id"]: a for a in body["items"][0]["algorithms"]}
    assert algorithms["linear_regression"]["available"] is True
    assert algorithms["linear_regression"]["task_types"] == ["regression"]
    assert algorithms["polynomial_regression"]["available"] is True
    assert algorithms["ridge"]["available"] is True
    assert algorithms["lasso"]["available"] is True
    assert algorithms["elastic_net"]["available"] is True
    assert algorithms["logistic_regression"]["available"] is True
    assert algorithms["logistic_regression"]["task_types"] == ["binary", "multiclass"]
    assert algorithms["sgd_regressor"]["task_types"] == ["regression"]
    assert algorithms["sgd_classifier"]["task_types"] == ["binary", "multiclass"]
    upcoming = {a["id"]: a for a in body["items"][1]["algorithms"]}
    assert upcoming["knn"] == {
        "id": "knn",
        "name": "K-Nearest Neighbors",
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
    assert body["resources"][0]["url"].startswith("https://")
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


def test_algorithm_learning_path_reaches_the_foundations(client: TestClient) -> None:
    body = client.get("/api/v1/algorithms/linear_regression").json()
    path = [c["id"] for c in body["learning_path"]]

    assert {c["id"] for c in body["concepts"]} <= set(path)
    assert len(path) == len(set(path))
    assert path[0] == "numpy-basics"
    assert path.index("matrix-multiplication") < path.index("normal-equation")
    assert path.index("derivatives") < path.index("gradient-descent")


def test_concept_index_groups_every_concept_by_area(client: TestClient) -> None:
    body = client.get("/api/v1/concepts").json()
    areas = {a["id"]: [c["id"] for c in a["concepts"]] for a in body["items"]}

    assert list(areas) == [
        "foundations",
        "optimisation",
        "evaluation",
        "features",
        "neural-networks",
        "sequence-models",
    ]
    assert areas["neural-networks"].index("neurons") < areas["neural-networks"].index(
        "backpropagation"
    )
    sequence = areas["sequence-models"]
    assert sequence.index("tokenization") < sequence.index("self-attention")
    assert sequence.index("causal-masking") < sequence.index("multi-head-attention")
    assert sequence[-1] == "transformer"  # everything else on the path comes first
    assert areas["foundations"][0] == "numpy-basics"
    foundations = areas["foundations"]
    assert foundations.index("vectors") < foundations.index("matrix-multiplication")
    every = [i for ids in areas.values() for i in ids]
    concept_dirs = Path(__file__).parents[3] / "content" / "concepts"
    assert sorted(every) == sorted(d.name for d in concept_dirs.iterdir())  # each exactly once
    mse = next(c for a in body["items"] for c in a["concepts"] if c["id"] == "mean-squared-error")
    assert mse["summary"]
    assert {"id": "linear_regression", "name": "Linear Regression"} in mse["algorithms"]


def test_concept_code_bundle_unzips_and_runs(client: TestClient, tmp_path: Path) -> None:
    response = client.get("/api/v1/concepts/feature-expansion/code/numpy/bundle")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="feature_expansion_numpy.zip"' in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == [
            "feature_expansion_numpy/README.md",
            "feature_expansion_numpy/feature_expansion_numpy.py",
            "feature_expansion_numpy/mean_squared_error_numpy.py",
            "feature_expansion_numpy/normal_equation_numpy.py",
        ]
        archive.extractall(tmp_path)
    folder = tmp_path / "feature_expansion_numpy"
    assert "python feature_expansion_numpy.py" in (folder / "README.md").read_text()
    # Exactly the code shown on the page, runnable with only the bundled imports.
    page = client.get("/api/v1/concepts/feature-expansion").json()["code"][0]["code"]
    assert (folder / "feature_expansion_numpy.py").read_text() == page
    run = subprocess.run(
        [sys.executable, "-E", "-s", "feature_expansion_numpy.py"],
        cwd=folder,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize(
    ("path", "status"),
    [("nope/code/numpy/bundle", 404), ("vectors/code/jax/bundle", 422)],
)
def test_concept_code_bundle_errors(client: TestClient, path: str, status: int) -> None:
    assert client.get(f"/api/v1/concepts/{path}").status_code == status


def test_concept_page_links_algorithms_and_prerequisites(client: TestClient) -> None:
    body = client.get("/api/v1/concepts/normal-equation").json()
    assert body["title"] == "Normal equation"
    assert [p["id"] for p in body["prerequisites"]] == [
        "mean-squared-error",
        "matrix-multiplication",
        "derivatives",
    ]
    # Every path reaches the foundations, each step after what it needs.
    assert [c["id"] for c in body["learning_path"]] == [
        "numpy-basics",
        "loss-function",
        "vectors",
        "mean-squared-error",
        "matrix-multiplication",
        "derivatives",
    ]
    assert all(r["note"] for r in body["resources"])
    assert body["algorithms"] == [
        {"id": "linear_regression", "name": "Linear Regression"},
        {"id": "polynomial_regression", "name": "Polynomial Regression"},
        {"id": "ridge", "name": "Ridge"},
    ]
    assert [(c["library"], c["title"], c["filename"]) for c in body["code"]] == [
        ("numpy", "NumPy", "normal_equation_numpy.py"),
        ("pytorch", "PyTorch", "normal_equation_pytorch.py"),
    ]
    numpy = body["code"][0]
    solve = next(a for a in numpy["annotations"] if a["anchor"] == "solve")
    assert "@a:" not in numpy["code"]
    span = numpy["code"].splitlines()[solve["start_line"] - 1 : solve["end_line"]]
    assert any("np.linalg.solve" in line for line in span)
    assert {"id": "gradient-descent", "title": "Gradient descent"} in solve["concepts"]


def test_unknown_ids_are_404_problems(client: TestClient) -> None:
    for path in (
        "/api/v1/algorithms/knn",
        "/api/v1/algorithms/linear_regression/code/nope",
        "/api/v1/concepts/nope",
        "/api/v1/tracks/nope",
        "/api/v1/papers/nope",
    ):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.headers["content-type"].startswith("application/problem+json")


def test_concept_code_lists_what_it_imports_from_prerequisites(client: TestClient) -> None:
    body = client.get("/api/v1/concepts/feature-expansion").json()
    numpy = next(c for c in body["code"] if c["library"] == "numpy")

    assert numpy["uses"] == [
        {
            "concept": {"id": "mean-squared-error", "title": "Mean squared error (MSE)"},
            "filename": "mean_squared_error_numpy.py",
            "names": ["mse"],
        },
        {
            "concept": {"id": "normal-equation", "title": "Normal equation"},
            "filename": "normal_equation_numpy.py",
            "names": ["least_squares"],
        },
    ]
    assert "from normal_equation_numpy import least_squares" in numpy["code"]
    path = [c["id"] for c in body["learning_path"]]
    assert path.index("mean-squared-error") < path.index("normal-equation")


def test_transformers_track_walks_every_prerequisite_to_the_transformer(
    client: TestClient,
) -> None:
    summary, rnn = client.get("/api/v1/tracks").json()["items"]
    assert summary["id"] == "transformers"
    families = {f["id"]: f for f in client.get("/api/v1/families").json()["items"]}
    assert families["sequence-models"]["tracks"] == [summary, rnn]
    assert families["sequence-models"]["algorithms"] == []
    assert families["linear-models"]["tracks"] == []
    assert summary["goal"] == {"id": "transformer", "title": "Transformer"}

    track = client.get("/api/v1/tracks/transformers").json()
    route = [c["id"] for chapter in track["chapters"] for c in chapter["concepts"]]
    path = client.get("/api/v1/concepts/transformer").json()["learning_path"]
    assert sorted(route) == sorted([c["id"] for c in path] + ["transformer"])
    assert summary["concept_count"] == len(route)
    assert route[-1] == "transformer"
    assert "::figure[attention-heatmap]" in track["body"]
    assert track["resources"]


def test_recurrent_networks_track_ends_at_the_lstm(client: TestClient) -> None:
    track = client.get("/api/v1/tracks/recurrent-networks").json()
    assert track["goal"] == {"id": "lstm", "title": "LSTM"}
    route = [c["id"] for chapter in track["chapters"] for c in chapter["concepts"]]
    path = client.get("/api/v1/concepts/lstm").json()["learning_path"]
    assert sorted(route) == sorted([c["id"] for c in path] + ["lstm"])
    assert route[-2:] == ["recurrent-neural-networks", "lstm"]
    assert "self-attention" not in route  # RNNs come before attention, not after


def test_attention_paper_walks_through_the_original_section_by_section(client: TestClient) -> None:
    summary, *_ = client.get("/api/v1/papers").json()["items"]
    assert summary["id"] == "attention-is-all-you-need"
    assert summary["source"]["type"] == "paper"
    assert summary["source"]["url"] == "https://arxiv.org/abs/1706.03762"

    paper = client.get("/api/v1/papers/attention-is-all-you-need").json()
    assert summary["section_count"] == len(paper["sections"])
    ids = [s["id"] for s in paper["sections"]]
    assert "3-2-1-scaled-dot-product-attention" in ids
    assert ids.index("3-2-1-scaled-dot-product-attention") < ids.index(
        "3-5-positional-encoding-where-is-each-word"
    )
    assert paper["track"]["id"] == "transformers"
    concepts = [c["id"] for c in paper["concepts"]]
    assert (
        concepts.index("softmax") < concepts.index("self-attention") < concepts.index("transformer")
    )
    # Every video the text plays is listed, uploaded or not.
    videos = {v["id"] for v in paper["videos"]}
    body = "\n".join(s["body"] for s in paper["sections"])
    assert all(f"::video[{v}]" in body for v in videos)
    assert paper["resources"]


def test_blog_walkthrough_credits_the_post_it_is_derived_from(client: TestClient) -> None:
    paper = client.get("/api/v1/papers/neural-networks-are-already-clean").json()
    assert paper["source"]["type"] == "blog"
    assert paper["source"]["url"] == "https://already-clean.goodfire.com/"
    assert "derived from" in paper["intro"] and paper["source"]["url"] in paper["intro"]
