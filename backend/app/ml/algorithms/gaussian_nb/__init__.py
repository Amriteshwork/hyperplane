"""Gaussian Naive Bayes: fit a bell curve per class and feature, then combine them with Bayes."""

from app.ml.algorithms.gaussian_nb.pack import GaussianNBPack

PACK = GaussianNBPack()

__all__ = ["PACK"]
