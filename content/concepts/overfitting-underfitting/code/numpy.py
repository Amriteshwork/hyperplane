"""Overfitting in NumPy: fit polynomials of rising degree, and watch train vs test error."""

import numpy as np

# @a:reuse
from mean_squared_error_numpy import mse

# @a:data
rng = np.random.default_rng(1)
x = np.concatenate([np.linspace(-1, 1, 20), rng.uniform(-1, 1, size=200)])
y = np.sin(3 * x) + rng.normal(scale=0.2, size=len(x))  # a smooth curve plus noise
train, test = np.arange(20), np.arange(20, len(x))  # only 20 training rows: easy to overfit


# @a:fit
def errors(degree):
    A = np.vander(x, degree + 1)  # columns xᵈ, …, x², x, 1
    beta = np.linalg.lstsq(A[train], y[train], rcond=None)[0]
    return [mse(y[rows], A[rows] @ beta) for rows in (train, test)]


# @a:sweep
results = {degree: errors(degree) for degree in (1, 4, 15)}
for degree, (train_mse, test_mse) in results.items():
    print(f"degree {degree:2}: train MSE {train_mse:.3f}   test MSE {test_mse:.3f}")

# @a:checks
under, good, over = results[1], results[4], results[15]
assert under[0] > 0.1 and under[1] > 0.1  # underfit: bad everywhere
assert good[1] < 0.1  # about the noise level (0.2² = 0.04)
assert over[0] < good[0]  # more flexible always fits the training rows better...
assert over[1] > 2 * good[1]  # ...but does worse on rows it never saw
