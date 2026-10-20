"""Feature expansion in PyTorch: a straight-line model fits a curve once we add an x² column."""

import torch

# @a:reuse
from mean_squared_error_pytorch import mse
from normal_equation_pytorch import least_squares

# @a:data
torch.manual_seed(0)
n = 200
x = torch.rand(n, dtype=torch.float64) * 4 - 2
y = 1.0 + 2.0 * x - 3.0 * x**2 + 0.3 * torch.randn(n, dtype=torch.float64)  # parabola + noise


# @a:fit
def fit_mse(features):
    A = torch.stack([torch.ones(n, dtype=torch.float64), *features], dim=1)  # 1s: intercept
    beta = least_squares(A, y)
    return beta, mse(y, A @ beta)


# @a:compare
line, line_mse = fit_mse([x])  # model: β₀ + β₁x
curve, curve_mse = fit_mse([x, x**2])  # same fit, one extra column: β₀ + β₁x + β₂x²
print(f"straight line:  MSE = {line_mse:.3f}")
print(f"with x² column: MSE = {curve_mse:.3f}, coefficients {curve.round(decimals=2).tolist()}")

# @a:checks
assert line_mse > 10 * curve_mse  # the line can't bend, so it misses badly
assert torch.allclose(curve, torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64), atol=0.1)
assert abs(curve_mse - 0.3**2) < 0.03  # what remains is just the noise
