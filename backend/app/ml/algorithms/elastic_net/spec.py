from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="elastic_net",
    name="Elastic Net",
    family="linear-models",
    task_types=frozenset({TaskType.REGRESSION}),
    difficulty=3,
    concepts=(
        "combined-penalties",
        "hyperparameter-search",
        "l1-regularization",
        "l2-regularization",
        "sparsity",
        "feature-selection",
        "regularization",
        "feature-scaling",
        "cross-validation",
        "bias-variance",
        "mean-squared-error",
        "r-squared",
        "train-test-split",
    ),
    supports=frozenset({Capability.COEFFICIENTS}),
    max_rows_default=1_000_000,
    content_path="content/algorithms/elastic_net",
)
