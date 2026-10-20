from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="ridge",
    name="Ridge",
    family="linear-models",
    task_types=frozenset({TaskType.REGRESSION}),
    difficulty=2,
    concepts=(
        "regularization",
        "l2-regularization",
        "feature-scaling",
        "cross-validation",
        "bias-variance",
        "normal-equation",
        "mean-squared-error",
        "r-squared",
        "train-test-split",
    ),
    supports=frozenset({Capability.COEFFICIENTS}),
    max_rows_default=1_000_000,
    content_path="content/algorithms/ridge",
)
