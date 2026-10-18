from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="linear_regression",
    name="Linear Regression",
    family="linear-models",
    task_types=frozenset({TaskType.REGRESSION}),
    difficulty=1,
    concepts=(
        "loss-function",
        "mean-squared-error",
        "normal-equation",
        "gradient-descent",
        "r-squared",
        "residuals",
        "linear-regression-assumptions",
        "train-test-split",
    ),
    supports=frozenset({Capability.COEFFICIENTS}),
    max_rows_default=1_000_000,
    content_path="content/algorithms/linear_regression",
)
