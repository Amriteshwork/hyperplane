"""Learning rates in NumPy: the same gradient descent with three step sizes, three outcomes."""

import numpy as np


# @a:limit
def largest_stable_rate(X):
    """Gradient descent on MSE diverges above 2 / (largest curvature) = n / λ_max(XᵀX)."""
    return len(X) / np.linalg.eigvalsh(X.T @ X).max()


# @a:decay
def inverse_scaling(eta0, t, power_t=0.5):
    """η₀ / t^power_t: big steps first, then ever smaller ones so noisy updates can settle."""
    return eta0 / t**power_t


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    n = 500
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])  # a 1s column for the intercept
    y = X @ np.array([4.0, 2.0, -3.0]) + rng.normal(scale=0.5, size=n)
    limit = largest_stable_rate(X)

    # @a:compare
    final = {}
    for factor in (0.002, 0.5, 1.05):
        eta, theta = factor * limit, np.zeros(3)
        for step in range(100):
            theta -= eta * 2 * X.T @ (X @ theta - y) / n  # one gradient-descent step on MSE
        final[factor] = np.mean((X @ theta - y) ** 2)
        print(f"η = {eta:.3f} ({factor:5} × limit): MSE after 100 steps {final[factor]:.4g}")
    assert final[0.5] < 0.3 < final[0.002]  # just right beats too small...
    assert final[1.05] > np.mean(y**2)  # ...and too large ends worse than where it started

    # @a:schedule
    jitter = {}
    for name, rate in (("constant", lambda t: 0.05), ("decaying", lambda t: inverse_scaling(0.5, t))):
        theta, path = np.zeros(3), []
        for t, i in enumerate(rng.integers(n, size=20_000), start=1):  # one random row per step
            theta -= rate(t) * 2 * X[i] * (X[i] @ theta - y[i])
            path.append(theta.copy())
        jitter[name] = np.std(path[-2000:], axis=0).max()  # how much θ still wanders at the end
        print(f"{name:8} rate: θ = {theta.round(3)}, still wandering by ±{jitter[name]:.4f}")
    assert jitter["decaying"] < jitter["constant"]
