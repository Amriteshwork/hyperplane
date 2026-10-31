"""KNN regression from scratch with NumPy: store the rows, then average the nearest targets."""

import numpy as np  # @a:numpy


# @a:distances
def pairwise_distances(A: np.ndarray, B: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """The distance from every row of A (one row per result) to every row of B (one column)."""
    if metric == "cosine":
        unit_A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
        unit_B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-300)
        return np.clip(1 - unit_A @ unit_B.T, 0.0, 2.0)
    differences = np.abs(A[:, None, :] - B[None, :, :])  # shape (rows of A, rows of B, features)
    if metric == "manhattan":
        return differences.sum(axis=2)
    return np.sqrt((differences**2).sum(axis=2))


# @a:class
class KNNRegressorScratch:
    def __init__(
        self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "euclidean"
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_: np.ndarray = np.empty((0, 0))
        self.y_: np.ndarray = np.empty(0)

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNRegressorScratch:
        self.X_, self.y_ = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        return self

    # @a:neighbours
    def kneighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each query's k smallest distances, and the training rows they belong to."""
        distances = pairwise_distances(np.asarray(X, dtype=float), self.X_, self.metric)
        rows = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        return np.take_along_axis(distances, rows, axis=1), rows

    # @a:weights
    def _weights(self, distances: np.ndarray) -> np.ndarray:
        if self.weights == "uniform":
            return np.ones_like(distances)
        exact = distances == 0
        with np.errstate(divide="ignore"):
            inverse = 1 / distances
        # A query identical to a training row takes its answer from such rows alone.
        return np.where(exact.any(axis=1, keepdims=True), exact.astype(float), inverse)

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        distances, rows = self.kneighbors(X)
        weights = self._weights(distances)
        return (weights * self.y_[rows]).sum(axis=1) / weights.sum(axis=1)
