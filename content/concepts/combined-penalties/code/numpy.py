"""Combined penalties in NumPy: L1 drops features, L2 keeps correlated ones together."""

import numpy as np

# @a:reuse
from l1_regularization_numpy import lasso, soft_threshold


# @a:elastic-net
def elastic_net(X, y, alpha, l1_ratio=0.5, tol=1e-10, max_sweeps=1000):
    """w minimising ‖y − Xw‖²/2n + αρ‖w‖₁ + α(1 − ρ)‖w‖²/2, ρ = l1_ratio. Centre X and y first."""
    n, p = X.shape
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    w = np.zeros(p)
    for _ in range(max_sweeps):
        previous = w.copy()
        for j in range(p):
            partial = y - X @ w + X[:, j] * w[j]  # the residual without feature j
            w[j] = soft_threshold(X[:, j] @ partial / n, l1) / (X[:, j] @ X[:, j] / n + l2)
        if np.max(np.abs(w - previous)) < tol:  # a whole sweep changed nothing: done
            break
    return w


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(size=n)
    group = signal[:, None] + 0.1 * rng.normal(size=(n, 3))  # three near-copies of one signal
    X = np.hstack([group, rng.normal(size=(n, 5))])  # ...plus five useless features
    y = 3 * signal + rng.normal(size=n)
    X, y = X - X.mean(axis=0), y - y.mean()  # centred, so there is no intercept to fit

    # @a:fit
    l1_part, l2_part = 0.5, 0.5
    only_l1 = lasso(X, y, l1_part)
    alpha = l1_part + l2_part  # same L1 part as the Lasso, plus an L2 part
    mixed = elastic_net(X, y, alpha, l1_ratio=l1_part / alpha)
    print("lasso:      ", only_l1.round(2))
    print("elastic net:", mixed.round(2))

    # @a:optimality
    kept = mixed != 0
    pull = X.T @ (y - X @ mixed) / len(y) - l2_part * mixed  # data's pull minus the L2 pull
    assert np.allclose(pull[kept], l1_part * np.sign(mixed[kept]))  # kept: balanced by L1
    assert np.all(np.abs(pull[~kept]) <= l1_part)  # dropped: not worth the L1 price

    # @a:checks
    assert np.all(only_l1[3:] == 0) and np.all(mixed[3:] == 0)  # both drop the noise...
    assert np.all(mixed[:3] > 0)  # ...Elastic Net keeps the whole group...
    assert np.ptp(mixed[:3]) < np.ptp(only_l1[:3]) / 3  # ...and shares its weight evenly
