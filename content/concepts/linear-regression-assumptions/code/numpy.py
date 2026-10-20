"""Linear regression assumptions in NumPy: check constant variance and multicollinearity."""

import numpy as np

# @a:setup
rng = np.random.default_rng(0)
n = 1000


# @a:fit
def fit(X, y):
    """Least-squares predictions with an intercept."""
    X1 = np.column_stack([np.ones(len(y)), X])
    return X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]


# @a:spread
# Homoscedasticity: the residual spread should be the same for small and large predictions.
def spread_ratio(fitted, residuals):
    """Residual spread of the top half of predictions over the bottom half; ≈ 1 if constant."""
    high = fitted > np.median(fitted)
    return residuals[high].std() / residuals[~high].std()


# @a:data
x = rng.uniform(0, 10, size=n)
constant = 2 * x + rng.normal(scale=1.0, size=n)  # noise the same everywhere
growing = 2 * x + rng.normal(scale=0.3 * x)  # noise grows with x: violates the assumption
# @a:check
ratios = {}
for name, y in [("constant noise", constant), ("growing noise", growing)]:
    fitted = fit(x, y)
    ratios[name] = spread_ratio(fitted, y - fitted)
    print(f"{name:>14}: residual spread ratio {ratios[name]:.2f}")
assert 0.8 < ratios["constant noise"] < 1.2
assert ratios["growing noise"] > 2


# @a:collinear
# Multicollinearity: if a feature is nearly a copy of another, its variance inflation factor
# VIF = 1 / (1 − R² of that feature predicted from the others) explodes.
a = rng.normal(size=n)
b = a + rng.normal(scale=0.05, size=n)  # almost the same as a
c = rng.normal(size=n)  # independent
features = np.column_stack([a, b, c])
# @a:vif
vif = {}
for j, name in enumerate("abc"):
    target = features[:, j]
    residual = target - fit(np.delete(features, j, axis=1), target)
    vif[name] = 1 / (residual.var() / target.var())  # 1 − R² = residual / total variance
    print(f"VIF({name}) = {vif[name]:.1f}")
assert vif["a"] > 100 and vif["b"] > 100  # their coefficients can't be trusted
assert vif["c"] < 1.1
