"""Lasso regression: least squares with an L1 penalty that sets some coefficients exactly to 0."""

from app.ml.algorithms.lasso.pack import LassoPack

PACK = LassoPack()

__all__ = ["PACK"]
