from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="lasso",
    name="Lasso",
    family="linear-models",
    task_types=frozenset({TaskType.REGRESSION}),
    difficulty=2,
    concepts=(
        "l1-regularization",
        "sparsity",
        "feature-selection",
        "regularization",
        "l2-regularization",
        "feature-scaling",
        "cross-validation",
        "bias-variance",
        "mean-squared-error",
        "r-squared",
        "train-test-split",
    ),
    supports=frozenset({Capability.COEFFICIENTS}),
    max_rows_default=1_000_000,
    content_path="content/algorithms/lasso",
)
