from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="polynomial_regression",
    name="Polynomial Regression",
    family="linear-models",
    task_types=frozenset({TaskType.REGRESSION}),
    difficulty=2,
    concepts=(
        "feature-expansion",
        "overfitting-underfitting",
        "bias-variance",
        "mean-squared-error",
        "normal-equation",
        "r-squared",
        "residuals",
        "train-test-split",
    ),
    supports=frozenset({Capability.COEFFICIENTS}),
    max_rows_default=1_000_000,
    content_path="content/algorithms/polynomial_regression",
)
