"""Normal equation in NumPy: the least-squares coefficients in one step, no iterations."""

import numpy as np

# @a:data
rng = np.random.default_rng(0)
n = 500
X = rng.normal(size=(n, 3))
true_beta = np.array([3.0, 2.0, -1.0, 0.5])  # intercept, then one weight per feature
X1 = np.column_stack([np.ones(n), X])  # a column of 1s gives the model its intercept
y = X1 @ true_beta + rng.normal(scale=0.1, size=n)

# @a:solve
# β = (XᵀX)⁻¹ Xᵀy. Solve the linear system instead of inverting: faster and more stable.
beta = np.linalg.solve(X1.T @ X1, X1.T @ y)
print("estimated:", beta.round(3))
print("true:     ", true_beta)
assert np.allclose(beta, true_beta, atol=0.05)

# @a:lstsq
# lstsq (via SVD) gives the same answer and also copes with collinear features.
assert np.allclose(beta, np.linalg.lstsq(X1, y, rcond=None)[0])
