from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="knn_classifier",
    name="KNN Classifier",
    family="instance-based",
    task_types=frozenset({TaskType.BINARY, TaskType.MULTICLASS}),
    difficulty=1,
    concepts=(
        "lazy-learning",
        "distance-metrics",
        "curse-of-dimensionality",
        "feature-scaling",
        "decision-boundary",
        "bias-variance",
        "overfitting-underfitting",
        "cross-validation",
        "classification-metrics",
        "train-test-split",
    ),
    supports=frozenset({Capability.PREDICT_PROBA, Capability.TUNING_CURVE, Capability.NEIGHBOURS}),
    # Fitting is cheap; each prediction searches the stored rows, and the model holds them all.
    max_rows_default=100_000,
    content_path="content/algorithms/knn_classifier",
)
