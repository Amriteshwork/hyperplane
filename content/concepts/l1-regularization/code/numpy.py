"""L1 regularisation (Lasso) in NumPy: a penalty that sets weights exactly to zero."""

import numpy as np

# @a:reuse
from normal_equation_numpy import least_squares


# @a:soft-threshold
def soft_threshold(z, t):
    """Move z towards 0 by t, and stop at 0."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


# @a:lasso
def lasso(X, y, alpha, tol=1e-10, max_sweeps=1000):
    """w minimising ‖y − Xw‖²/2n + α‖w‖₁, one weight at a time. Centre X and y first."""
    n, p = X.shape
    w = np.zeros(p)
    for _ in range(max_sweeps):
        previous = w.copy()
        for j in range(p):
            partial = y - X @ w + X[:, j] * w[j]  # the residual without feature j
            w[j] = soft_threshold(X[:, j] @ partial / n, alpha) / (X[:, j] @ X[:, j] / n)
        if np.max(np.abs(w - previous)) < tol:  # a whole sweep changed nothing: done
            break
    return w


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    n, p = 200, 8
    X = rng.normal(size=(n, p))
    y = X @ np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0]) + rng.normal(size=n)  # 3 features matter
    X, y = X - X.mean(axis=0), y - y.mean()  # centred, so there is no intercept to fit

    # @a:fit
    alpha = 0.25
    ols = least_squares(X, y)
    l1 = lasso(X, y, alpha)
    l2 = np.linalg.solve(X.T @ X + 50.0 * np.eye(p), X.T @ y)  # Ridge, for comparison
    print("least squares:", ols.round(3))
    print("lasso (L1):   ", l1.round(3))
    print("ridge (L2):   ", l2.round(3))

    # @a:optimality
    correlation = X.T @ (y - X @ l1) / n  # how much each feature could still explain
    assert np.allclose(correlation[l1 != 0], alpha * np.sign(l1[l1 != 0]))  # kept: exactly ±α
    assert np.all(np.abs(correlation[l1 == 0]) <= alpha)  # dropped: not worth α

    # @a:checks
    assert np.all(l1[3:] == 0) and np.all(l1[:3] != 0)  # exactly the useless features are 0
    assert np.all(l2 != 0)  # Ridge shrinks everything, but zeroes nothing
    assert np.all(np.abs(l1[:3]) < np.abs(ols[:3]))  # the kept weights are shrunk, too
