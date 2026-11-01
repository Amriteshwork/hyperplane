from app.ml.base import AlgorithmSpec, Capability, TaskType

SPEC = AlgorithmSpec(
    id="gaussian_nb",
    name="Gaussian Naive Bayes",
    family="probabilistic",
    task_types=frozenset({TaskType.BINARY, TaskType.MULTICLASS}),
    difficulty=2,
    concepts=(
        "bayes-theorem",
        "likelihood",
        "conditional-independence",
        "probability",
        "logarithms-exponentials",
        "log-loss",
        "decision-boundary",
        "cross-validation",
        "classification-metrics",
        "train-test-split",
    ),
    supports=frozenset(
        {Capability.PREDICT_PROBA, Capability.TUNING_CURVE, Capability.CLASS_GAUSSIANS}
    ),
    # Fitting is one pass over the data, but it needs a dense matrix, so one-hot columns cost memory.
    max_rows_default=200_000,
    dense_input=True,
    content_path="content/algorithms/gaussian_nb",
)
