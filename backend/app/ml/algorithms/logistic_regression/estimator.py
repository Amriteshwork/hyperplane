from sklearn.linear_model import LogisticRegression

from app.ml.algorithms.logistic_regression.params import LogisticRegressionParams


def build_estimator(params: LogisticRegressionParams) -> LogisticRegression:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return LogisticRegression(**params.model_dump())
