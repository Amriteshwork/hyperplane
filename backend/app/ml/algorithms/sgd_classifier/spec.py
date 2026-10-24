from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="sgd_classifier",
    name="SGD Classifier",
    family="linear-models",
    task_types=frozenset({TaskType.BINARY, TaskType.MULTICLASS}),
    difficulty=3,
    concepts=(
        "stochastic-gradient-descent",
        "learning-rate",
        "epochs",
        "gradient-descent",
        "log-loss",
        "sigmoid",
        "decision-boundary",
        "classification-metrics",
        "roc-auc",
        "class-imbalance",
        "l2-regularization",
        "feature-scaling",
        "train-test-split",
    ),
    # Probabilities only with the log_loss or modified_huber loss; the trainer checks.
    supports=frozenset({Capability.PREDICT_PROBA, Capability.CLASS_COEFFICIENTS}),
    max_rows_default=1_000_000,
    content_path="content/algorithms/sgd_classifier",
)
