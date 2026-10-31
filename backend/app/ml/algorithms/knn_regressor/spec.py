from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="knn_regressor",
    name="KNN Regressor",
    family="instance-based",
    task_types=frozenset({TaskType.REGRESSION}),
    difficulty=1,
    concepts=(
        "lazy-learning",
        "distance-metrics",
        "curse-of-dimensionality",
        "feature-scaling",
        "bias-variance",
        "overfitting-underfitting",
        "cross-validation",
        "mean-squared-error",
        "r-squared",
        "train-test-split",
    ),
    supports=frozenset({Capability.TUNING_CURVE, Capability.NEIGHBOURS}),
    # Fitting is cheap; each prediction searches the stored rows, and the model holds them all.
    max_rows_default=100_000,
    content_path="content/algorithms/knn_regressor",
)
