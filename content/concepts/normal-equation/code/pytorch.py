"""Normal equation in PyTorch: the least-squares coefficients in one step, no iterations."""

import torch

# @a:data
torch.manual_seed(0)
n = 500
X = torch.randn(n, 3, dtype=torch.float64)
true_beta = torch.tensor([3.0, 2.0, -1.0, 0.5], dtype=torch.float64)  # intercept, then weights
X1 = torch.cat([torch.ones(n, 1, dtype=torch.float64), X], dim=1)  # 1s give the intercept
y = X1 @ true_beta + 0.1 * torch.randn(n, dtype=torch.float64)

# @a:solve
# β = (XᵀX)⁻¹ Xᵀy. Solve the linear system instead of inverting: faster and more stable.
beta = torch.linalg.solve(X1.T @ X1, X1.T @ y)
print("estimated:", beta.numpy().round(3))
print("true:     ", true_beta.numpy())
assert torch.allclose(beta, true_beta, atol=0.05)

# @a:lstsq
# lstsq gives the same answer and also copes with collinear features.
assert torch.allclose(beta, torch.linalg.lstsq(X1, y.unsqueeze(1)).solution.squeeze(1))
