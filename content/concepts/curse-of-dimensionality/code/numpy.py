"""The curse of dimensionality in NumPy: distances concentrate, and neighbours stop being near."""

import numpy as np

# @a:reuse
from distance_metrics_numpy import pairwise
from lazy_learning_numpy import knn_classify


# @a:contrast
def relative_contrast(X, query):
    """(farthest − nearest) / nearest: how much better the nearest row is than the farthest."""
    d = pairwise(query[None, :], X)[0]
    return (d.max() - d.min()) / d.min()


# @a:edge
def edge_for_fraction(fraction, d):
    """The side of a sub-cube of the unit cube that holds this fraction of its volume."""
    return fraction ** (1 / d)


# @a:demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # @a:concentration
    contrast = {}
    for d in (2, 10, 100, 1000):
        X = rng.uniform(size=(1000, d))
        contrast[d] = np.mean([relative_contrast(X, q) for q in rng.uniform(size=(20, d))])
        print(f"d = {d:4d}: farthest is {1 + contrast[d]:7.2f} × as far as nearest")
    assert contrast[2] > 20 * contrast[100] > 20 * contrast[1000]
    assert contrast[1000] < 0.3

    # @a:neighbourhood
    for d in (1, 2, 10, 100):
        print(f"d = {d:3d}: a cube holding 1% of the data has sides {edge_for_fraction(0.01, d):.2f}")
    assert edge_for_fraction(0.01, 1) == 0.01
    assert edge_for_fraction(0.01, 100) > 0.95  # "local" now spans almost every feature's range

    # @a:rows-needed
    rows_at_one_tenth = {d: 10**d for d in (1, 2, 3, 6)}  # 10 bins per axis, one row in each
    print("rows to keep a neighbour within 0.1 on each axis:", rows_at_one_tenth)

    # @a:noise
    def signal_and_noise(n, noise_features):
        y = rng.integers(0, 2, size=n)
        signal = rng.normal(size=(n, 2)) + 2.0 * y[:, None]  # only these two features matter
        return np.column_stack([signal, rng.normal(size=(n, noise_features))]), y

    accuracy = {}
    for noise_features in (0, 10, 100, 1000):
        X_train, y_train = signal_and_noise(300, noise_features)
        X_test, y_test = signal_and_noise(300, noise_features)
        accuracy[noise_features] = (knn_classify(X_train, y_train, X_test, 15) == y_test).mean()
        print(f"{noise_features:4d} noise features: test accuracy {accuracy[noise_features]:.3f}")
    assert accuracy[0] > 0.9  # two informative features: an easy problem ...
    assert accuracy[1000] < 0.7  # ... buried by features that say nothing
