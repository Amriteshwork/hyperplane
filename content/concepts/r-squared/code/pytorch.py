"""R² in PyTorch: how much of the target's variance a model explains, compared with the mean."""

import torch

# @a:reuse
from mean_squared_error_pytorch import mse

# @a:data
torch.manual_seed(0)
n = 300
X = torch.randn(n, 2, dtype=torch.float64)
y = X @ torch.tensor([1.5, -2.0], dtype=torch.float64) + torch.randn(n, dtype=torch.float64)


# @a:formula
def r_squared(y, y_pred):
    model = mse(y, y_pred)  # error the model leaves
    mean_only = mse(y, torch.full_like(y, y.mean().item()))  # error of predicting the mean
    return 1 - model / mean_only  # SS_res / SS_tot: the 1/n cancels


# @a:fit
X1 = torch.cat([torch.ones(n, 1, dtype=torch.float64), X], dim=1)
y_pred = X1 @ torch.linalg.lstsq(X1, y.unsqueeze(1)).solution.squeeze(1)

# @a:compare
fitted = r_squared(y, y_pred)
mean_only = r_squared(y, torch.full_like(y, y.mean().item()))
shuffled = r_squared(y, y_pred[torch.randperm(n)])  # right values, wrong rows
print(f"fitted model:          R² = {fitted:.3f}")
print(f"always predict mean:   R² = {mean_only:.3f}")
print(f"shuffled predictions:  R² = {shuffled:.3f}")

# @a:checks
assert 0.8 < fitted < 1  # signal variance 6.25, noise 1 → about 0.86
assert mean_only.abs() < 1e-12  # the mean is the zero point
assert shuffled < 0  # worse than the mean is possible
# For least squares with an intercept, R² is the squared correlation of y and ŷ.
assert torch.isclose(fitted, torch.corrcoef(torch.stack([y, y_pred]))[0, 1] ** 2)
