"""L2 regularisation (Ridge) in NumPy: a solution even when least squares has none."""

import numpy as np

# @a:reuse
from normal_equation_numpy import least_squares

# @a:data
rng = np.random.default_rng(0)
n = 100
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(scale=0.01, size=n)  # an almost exact copy of x1
X = np.column_stack([x1, x2])
y = 3 * x1 + rng.normal(scale=0.5, size=n)  # only the shared signal matters

# @a:ols
ols = least_squares(X, y)

# @a:ridge
alpha = 1.0
ridge = np.linalg.solve(X.T @ X + alpha * np.eye(2), X.T @ y)  # (XᵀX + αI) w = Xᵀy
print("least squares:", ols.round(2))
print("ridge:        ", ridge.round(2))

# @a:conditioning
cond_ols = np.linalg.cond(X.T @ X)
cond_ridge = np.linalg.cond(X.T @ X + alpha * np.eye(2))
print(f"condition number: {cond_ols:,.0f} → {cond_ridge:,.0f}")


# @a:resample
def refit(rows):
    Xr, yr = X[rows], y[rows]
    return least_squares(Xr, yr), np.linalg.solve(Xr.T @ Xr + alpha * np.eye(2), Xr.T @ yr)


fits = [refit(rng.integers(0, n, size=n)) for _ in range(50)]  # 50 bootstrap samples
ols_spread = np.std([o[0] for o, _ in fits])
ridge_spread = np.std([r[0] for _, r in fits])
print(f"spread of w₁ across samples: OLS {ols_spread:.2f}, Ridge {ridge_spread:.3f}")

# @a:checks
assert ols_spread > 10 * ridge_spread  # OLS splits the credit differently every sample
assert abs(ridge[0] - ridge[1]) < 0.1  # Ridge shares it equally...
assert abs(ridge.sum() - 3) < 0.3  # ...and the shares add up to the true effect
assert np.linalg.norm(ridge) < np.linalg.norm(ols)  # smaller weights, same predictions
assert cond_ridge < cond_ols / 100  # adding α to the diagonal makes the solve stable
