"""KNN classification from scratch with NumPy: store the rows, then let the nearest ones vote."""

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
class KNNClassifierScratch:
    def __init__(
        self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "euclidean"
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_: np.ndarray = np.empty((0, 0))
        self.classes_: np.ndarray = np.empty(0)
        self.y_: np.ndarray = np.empty(0, dtype=int)

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifierScratch:
        self.X_ = np.asarray(X, dtype=float)
        self.classes_, self.y_ = np.unique(y, return_inverse=True)  # labels become 0, 1, 2 …
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

    # @a:predict-proba
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances, rows = self.kneighbors(X)
        votes = np.zeros((len(distances), len(self.classes_)))
        np.add.at(votes, (np.arange(len(rows))[:, None], self.y_[rows]), self._weights(distances))
        return votes / votes.sum(axis=1, keepdims=True)

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        # argmax takes the first of equal scores, so a tied vote goes to the earliest class.
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
