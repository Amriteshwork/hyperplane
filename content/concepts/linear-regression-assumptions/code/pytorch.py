"""Linear regression assumptions in PyTorch: check constant variance and multicollinearity."""

import torch

# @a:setup
torch.manual_seed(0)
n = 1000
f64 = torch.float64


# @a:fit
def fit(X, y):
    """Least-squares predictions with an intercept."""
    X1 = torch.cat([torch.ones(len(y), 1, dtype=f64), X.reshape(len(y), -1)], dim=1)
    return X1 @ torch.linalg.lstsq(X1, y.unsqueeze(1)).solution.squeeze(1)


# @a:spread
# Homoscedasticity: the residual spread should be the same for small and large predictions.
def spread_ratio(fitted, residuals):
    """Residual spread of the top half of predictions over the bottom half; ≈ 1 if constant."""
    high = fitted > fitted.median()
    return (residuals[high].std() / residuals[~high].std()).item()


# @a:data
x = torch.empty(n, dtype=f64).uniform_(0, 10)
constant = 2 * x + torch.randn(n, dtype=f64)  # noise the same everywhere
growing = 2 * x + 0.3 * x * torch.randn(n, dtype=f64)  # noise grows with x: a violation
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
a = torch.randn(n, dtype=f64)
b = a + 0.05 * torch.randn(n, dtype=f64)  # almost the same as a
c = torch.randn(n, dtype=f64)  # independent
features = torch.stack([a, b, c], dim=1)
# @a:vif
vif = {}
for j, name in enumerate("abc"):
    target = features[:, j]
    others = features[:, [k for k in range(3) if k != j]]
    residual = target - fit(others, target)
    vif[name] = (target.var() / residual.var()).item()  # 1 − R² = residual / total variance
    print(f"VIF({name}) = {vif[name]:.1f}")
assert vif["a"] > 100 and vif["b"] > 100  # their coefficients can't be trusted
assert vif["c"] < 1.1
