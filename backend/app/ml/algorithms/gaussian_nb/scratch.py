"""Gaussian Naive Bayes from scratch with NumPy: a bell curve per class and feature, then Bayes."""

import numpy as np  # @a:numpy


# @a:class
class GaussianNBScratch:
    def __init__(self, var_smoothing: float = 0.01) -> None:
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray = np.empty(0)
        self.class_prior_: np.ndarray = np.empty(0)
        self.theta_: np.ndarray = np.empty((0, 0))
        self.var_: np.ndarray = np.empty((0, 0))

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNBScratch:
        X = np.asarray(X, dtype=float)
        self.classes_, labels = np.unique(y, return_inverse=True)
        rows = [X[labels == c] for c in range(len(self.classes_))]
        self.class_prior_ = np.array([len(r) / len(X) for r in rows])
        self.theta_ = np.array([r.mean(axis=0) for r in rows])
        variances = np.array([r.var(axis=0) for r in rows])
        # @a:smoothing
        self.var_ = variances + self.var_smoothing * X.var(axis=0).max()
        return self

    # @a:joint
    def joint_log_proba(self, X: np.ndarray) -> np.ndarray:
        """log P(class) + Σ log p(feature | class), with one row per query and column per class."""
        X = np.asarray(X, dtype=float)[:, None, :]  # shape (queries, 1, features)
        log_bell = -0.5 * np.log(2 * np.pi * self.var_) - (X - self.theta_) ** 2 / (2 * self.var_)
        return np.log(self.class_prior_) + log_bell.sum(axis=2)

    # @a:predict-proba
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self.joint_log_proba(X)
        # Subtract the largest score first: exponentials of large negatives would all round to 0.
        scaled = np.exp(joint - joint.max(axis=1, keepdims=True))
        return scaled / scaled.sum(axis=1, keepdims=True)

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.joint_log_proba(X), axis=1)]
