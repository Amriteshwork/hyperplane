from sklearn.linear_model import ElasticNet

from app.ml.algorithms.elastic_net.params import ElasticNetParams


def build_estimator(params: ElasticNetParams) -> ElasticNet:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return ElasticNet(**params.model_dump())
