from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="mlp_regressor",
    name="MLP Regressor",
    family="neural-networks",
    task_types=frozenset({TaskType.REGRESSION}),
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
        "mean-squared-error",
        "l2-regularization",
        "overfitting-underfitting",
        "feature-scaling",
        "r-squared",
        "train-test-split",
    ),
    supports=frozenset(
        {Capability.TRAINING_CURVE, Capability.NETWORK, Capability.FEATURE_IMPORTANCE}
    ),
    max_rows_default=200_000,
    content_path="content/algorithms/mlp_regressor",
)
