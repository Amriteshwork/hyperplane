from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="logistic_regression",
    name="Logistic Regression",
    family="linear-models",
    task_types=frozenset({TaskType.BINARY, TaskType.MULTICLASS}),
    difficulty=2,
    concepts=(
        "sigmoid",
        "softmax",
        "log-loss",
        "decision-boundary",
        "classification-metrics",
        "roc-auc",
        "class-imbalance",
        "logarithms-exponentials",
        "gradient-descent",
        "l2-regularization",
        "feature-scaling",
        "train-test-split",
    ),
    supports=frozenset({Capability.PREDICT_PROBA, Capability.CLASS_COEFFICIENTS}),
    max_rows_default=1_000_000,
    content_path="content/algorithms/logistic_regression",
)
