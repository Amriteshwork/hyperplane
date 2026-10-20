"""Feature expansion in NumPy: a straight-line model fits a curve once we add an x² column."""

import numpy as np

# @a:reuse
from mean_squared_error_numpy import mse
from normal_equation_numpy import least_squares

# @a:data
rng = np.random.default_rng(0)
n = 200
x = rng.uniform(-2, 2, size=n)
y = 1.0 + 2.0 * x - 3.0 * x**2 + rng.normal(scale=0.3, size=n)  # a parabola plus noise


# @a:fit
def fit_mse(features):
    A = np.column_stack([np.ones(n), *features])  # a column of 1s for the intercept
    beta = least_squares(A, y)
    return beta, mse(y, A @ beta)


# @a:compare
line, line_mse = fit_mse([x])  # model: β₀ + β₁x
curve, curve_mse = fit_mse([x, x**2])  # same fit, one extra column: β₀ + β₁x + β₂x²
print(f"straight line:  MSE = {line_mse:.3f}")
print(f"with x² column: MSE = {curve_mse:.3f}, coefficients {curve.round(2)}")

# @a:checks
assert line_mse > 10 * curve_mse  # the line can't bend, so it misses badly
assert np.allclose(curve, [1.0, 2.0, -3.0], atol=0.1)  # recovers the true parabola
assert abs(curve_mse - 0.3**2) < 0.03  # what remains is just the noise
