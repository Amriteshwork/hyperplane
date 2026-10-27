from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="mlp_classifier",
    name="MLP Classifier",
    family="neural-networks",
    task_types=frozenset({TaskType.BINARY, TaskType.MULTICLASS}),
    difficulty=4,
    concepts=(
        "neurons",
        "activation-functions",
        "backpropagation",
        "adam-optimizer",
        "early-stopping",
        "stochastic-gradient-descent",
        "learning-rate",
        "epochs",
        "sigmoid",
        "softmax",
        "log-loss",
        "decision-boundary",
        "classification-metrics",
        "roc-auc",
        "l2-regularization",
        "overfitting-underfitting",
        "feature-scaling",
        "train-test-split",
    ),
    supports=frozenset(
        {
            Capability.PREDICT_PROBA,
            Capability.TRAINING_CURVE,
            Capability.NETWORK,
            Capability.FEATURE_IMPORTANCE,
        }
    ),
    max_rows_default=200_000,
    content_path="content/algorithms/mlp_classifier",
)
