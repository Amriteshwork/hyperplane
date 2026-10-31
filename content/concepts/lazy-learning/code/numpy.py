"""Lazy learning in NumPy: k-nearest neighbours, where fitting stores and predicting does the work."""

import numpy as np

# @a:reuse
from distance_metrics_numpy import pairwise


# @a:neighbours
def nearest(X_train, X_query, k):
    """The row numbers of each query's k closest training rows."""
    return np.argsort(pairwise(X_query, X_train), axis=1)[:, :k]


# @a:classify
def knn_classify(X_train, y_train, X_query, k):
    """The most common label among each query's k nearest rows; y_train holds 0, 1, 2 …"""
    votes = y_train[nearest(X_train, X_query, k)]
    return np.array([np.bincount(row).argmax() for row in votes])


# @a:regress
def knn_regress(X_train, y_train, X_query, k):
    """The average target of each query's k nearest rows."""
    return y_train[nearest(X_train, X_query, k)].mean(axis=1)


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)

    def two_blobs(n):
        y = rng.integers(0, 2, size=n)
        X = rng.normal(size=(n, 2)) + 1.2 * y[:, None]
        return X, y

    X_train, y_train = two_blobs(300)
    X_test, y_test = two_blobs(300)

    # @a:sweep
    print("  k  train  test")
    accuracy = {}
    for k in (1, 15, 300):
        train = (knn_classify(X_train, y_train, X_train, k) == y_train).mean()
        test = (knn_classify(X_train, y_train, X_test, k) == y_test).mean()
        accuracy[k] = (train, test)
        print(f"{k:3d}  {train:.3f}  {test:.3f}")
    assert accuracy[1][0] == 1.0  # k = 1 memorises: every training row is its own neighbour
    assert accuracy[15][1] > accuracy[1][1]  # averaging over noise generalises better
    assert accuracy[300][1] < 0.6  # k = n votes the majority class for every query

    # @a:cost
    n_train, n_queries = len(X_train), len(X_test)
    print(f"fit: stored {X_train.size} numbers, computed 0 distances")
    print(f"predict: {n_train * n_queries:,} distances for {n_queries} queries")
    distances = pairwise(X_test, X_train)
    assert distances.size == n_train * n_queries

    # @a:regression
    x = rng.uniform(-3, 3, size=(200, 1))
    y = np.sin(x[:, 0]) + rng.normal(scale=0.2, size=200)
    grid = np.linspace(-3, 3, 50)[:, None]
    smooth = knn_regress(x, y, grid, 10)
    print(f"regression MSE against sin(x), k = 10: {np.mean((smooth - np.sin(grid[:, 0])) ** 2):.4f}")
    assert np.mean((smooth - np.sin(grid[:, 0])) ** 2) < 0.03  # follows the curve, no formula given
    assert y.min() <= smooth.min() and smooth.max() <= y.max()  # averages can't leave the range
