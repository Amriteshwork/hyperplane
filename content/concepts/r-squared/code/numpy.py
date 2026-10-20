"""R² in NumPy: how much of the target's variance a model explains, compared with the mean."""

import numpy as np

# @a:reuse
from mean_squared_error_numpy import mse

# @a:data
rng = np.random.default_rng(0)
n = 300
X = rng.normal(size=(n, 2))
y = X @ np.array([1.5, -2.0]) + rng.normal(scale=1.0, size=n)


# @a:formula
def r_squared(y, y_pred):
    model = mse(y, y_pred)  # error the model leaves
    mean_only = mse(y, np.full_like(y, y.mean()))  # error of always predicting the mean
    return 1 - model / mean_only  # SS_res / SS_tot: the 1/n cancels


# @a:fit
X1 = np.column_stack([np.ones(n), X])
y_pred = X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]

# @a:compare
fitted = r_squared(y, y_pred)
mean_only = r_squared(y, np.full(n, y.mean()))
shuffled = r_squared(y, rng.permutation(y_pred))  # right values, wrong rows
print(f"fitted model:          R² = {fitted:.3f}")
print(f"always predict mean:   R² = {mean_only:.3f}")
print(f"shuffled predictions:  R² = {shuffled:.3f}")

# @a:checks
assert 0.8 < fitted < 1  # signal variance 6.25, noise 1 → about 0.86
assert np.isclose(mean_only, 0)  # the mean is the zero point
assert shuffled < 0  # worse than the mean is possible
# For least squares with an intercept, R² is the squared correlation of y and ŷ.
assert np.isclose(fitted, np.corrcoef(y, y_pred)[0, 1] ** 2)
