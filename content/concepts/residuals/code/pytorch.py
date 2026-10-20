"""Residuals in PyTorch: what's left of each target after the model's prediction."""

import torch

# @a:data
torch.manual_seed(0)
x = torch.empty(200, dtype=torch.float64).uniform_(-3, 3)  # float64: exact checks below
y = 1.5 * x + 0.5 * x**2 + 0.5 * torch.randn(200, dtype=torch.float64)  # the truth has a curve

# @a:fit
# Fit a straight line (intercept + slope) by least squares.
X = torch.stack([torch.ones_like(x), x], dim=1)
beta = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
# @a:residuals
residuals = y - X @ beta  # true value minus prediction, one per row

# @a:mean
print(f"mean residual: {residuals.mean():.1e}")  # ~0: the intercept absorbs any offset
assert residuals.mean().abs() < 1e-10

# @a:orthogonal
# Least squares leaves residuals uncorrelated with every feature it used...
assert (residuals @ x).abs() < 1e-8

# @a:pattern
# ...but not with what it missed. They still follow x², the curve a line can't bend to:
# the pattern a residual plot shows you.
pattern = torch.corrcoef(torch.stack([residuals, x**2]))[0, 1]
print(f"correlation of residuals with x²: {pattern:.2f}")
assert pattern > 0.5
