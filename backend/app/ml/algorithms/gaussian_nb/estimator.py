from sklearn.naive_bayes import GaussianNB

from app.ml.algorithms.gaussian_nb.params import GaussianNBParams


def build_estimator(params: GaussianNBParams) -> GaussianNB:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return GaussianNB(**params.model_dump())
