from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="sgd_regressor",
    name="SGD Regressor",
    family="linear-models",
    task_types=frozenset({TaskType.REGRESSION}),
    difficulty=3,
    concepts=(
        "stochastic-gradient-descent",
        "learning-rate",
        "epochs",
        "gradient-descent",
        "loss-function",
        "mean-squared-error",
        "l2-regularization",
        "combined-penalties",
        "feature-scaling",
        "r-squared",
        "train-test-split",
    ),
    supports=frozenset({Capability.COEFFICIENTS}),
    max_rows_default=1_000_000,
    content_path="content/algorithms/sgd_regressor",
)
