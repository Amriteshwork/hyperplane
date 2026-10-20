"""Residuals in NumPy: what's left of each target after the model's prediction."""

import numpy as np

# @a:data
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, size=200)
y = 1.5 * x + 0.5 * x**2 + rng.normal(scale=0.5, size=200)  # the truth has a curve

# @a:fit
# Fit a straight line (intercept + slope) by least squares.
X = np.column_stack([np.ones_like(x), x])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
# @a:residuals
residuals = y - X @ beta  # true value minus prediction, one per row

# @a:mean
print(f"mean residual: {residuals.mean():.1e}")  # ~0: the intercept absorbs any offset
assert abs(residuals.mean()) < 1e-10

# @a:orthogonal
# Least squares leaves residuals uncorrelated with every feature it used...
assert abs(residuals @ x) < 1e-8

# @a:pattern
# ...but not with what it missed. They still follow x², the curve a line can't bend to:
# the pattern a residual plot shows you.
pattern = np.corrcoef(residuals, x**2)[0, 1]
print(f"correlation of residuals with x²: {pattern:.2f}")
assert pattern > 0.5
