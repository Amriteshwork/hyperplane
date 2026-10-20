"""L2 regularisation (Ridge) in PyTorch: a solution even when least squares has none."""

import torch

# @a:reuse
from normal_equation_pytorch import least_squares

# @a:data
torch.manual_seed(0)
f64 = torch.float64
n = 100
x1 = torch.randn(n, dtype=f64)
x2 = x1 + 0.01 * torch.randn(n, dtype=f64)  # an almost exact copy of x1
X = torch.stack([x1, x2], dim=1)
y = 3 * x1 + 0.5 * torch.randn(n, dtype=f64)  # only the shared signal matters

# @a:ols
ols = least_squares(X, y)

# @a:ridge
alpha = 1.0
eye = torch.eye(2, dtype=f64)
ridge = torch.linalg.solve(X.T @ X + alpha * eye, X.T @ y)  # (XᵀX + αI) w = Xᵀy
print("least squares:", ols.round(decimals=2).tolist())
print("ridge:        ", ridge.round(decimals=2).tolist())

# @a:conditioning
cond_ols = torch.linalg.cond(X.T @ X).item()
cond_ridge = torch.linalg.cond(X.T @ X + alpha * eye).item()
print(f"condition number: {cond_ols:,.0f} → {cond_ridge:,.0f}")


# @a:resample
def refit(rows):
    Xr, yr = X[rows], y[rows]
    return least_squares(Xr, yr), torch.linalg.solve(Xr.T @ Xr + alpha * eye, Xr.T @ yr)


fits = [refit(torch.randint(0, n, (n,))) for _ in range(50)]  # 50 bootstrap samples
ols_spread = torch.stack([o[0] for o, _ in fits]).std(correction=0).item()
ridge_spread = torch.stack([r[0] for _, r in fits]).std(correction=0).item()
print(f"spread of w₁ across samples: OLS {ols_spread:.2f}, Ridge {ridge_spread:.3f}")

# @a:checks
assert ols_spread > 10 * ridge_spread  # OLS splits the credit differently every sample
assert abs(ridge[0] - ridge[1]) < 0.1  # Ridge shares it equally...
assert abs(ridge.sum() - 3) < 0.3  # ...and the shares add up to the true effect
assert torch.linalg.norm(ridge) < torch.linalg.norm(ols)  # smaller weights, same predictions
assert cond_ridge < cond_ols / 100  # adding α to the diagonal makes the solve stable
