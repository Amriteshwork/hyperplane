"""Elastic Net: least squares with both an L1 and an L2 penalty, mixed by `l1_ratio`."""

from app.ml.algorithms.elastic_net.pack import ElasticNetPack

PACK = ElasticNetPack()

__all__ = ["PACK"]
