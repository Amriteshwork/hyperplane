"""Log-loss in PyTorch: score probabilities by how likely they made the true labels."""

import torch

# @a:reuse
from sigmoid_pytorch import sigmoid


# @a:log-loss
def log_loss(y, p, eps=1e-15):
    """Mean of −[y log p + (1 − y) log(1 − p)]: the negative log-likelihood per row."""
    p = torch.clamp(torch.as_tensor(p, dtype=torch.float64), eps, 1 - eps)  # log(0) is −∞
    y = torch.as_tensor(y, dtype=torch.float64)
    return -torch.mean(y * torch.log(p) + (1 - y) * torch.log(1 - p))


# @a:demo
if __name__ == "__main__":
    f64 = torch.float64

    # @a:confidence
    for p in (0.9, 0.6, 0.4, 0.1, 0.01):
        print(f"true label 1, predicted {p:<4}: log-loss {log_loss(1.0, p).item():.3f}, "
              f"squared error {(1 - p) ** 2:.3f}")
    assert log_loss(1.0, 0.01) > 4 > (1 - 0.01) ** 2  # confident and wrong: punished hard

    # @a:likelihood
    y = torch.tensor([1, 0, 1, 1, 0], dtype=f64)
    p = torch.tensor([0.8, 0.3, 0.6, 0.9, 0.2], dtype=f64)
    likelihood = torch.prod(torch.where(y == 1, p, 1 - p))  # chance of exactly these labels
    assert torch.isclose(log_loss(y, p), -torch.log(likelihood) / len(y))  # = −log(likelihood)/n

    # @a:data
    torch.manual_seed(0)
    n = 500
    X = torch.randn(n, 2, dtype=f64)
    y = (torch.rand(n, dtype=f64) < sigmoid(X @ torch.tensor([2.0, -1.0], dtype=f64) + 0.5)).to(f64)

    # @a:gradient
    w = torch.tensor([0.3, 0.2], dtype=f64, requires_grad=True)
    b = torch.tensor(-0.1, dtype=f64, requires_grad=True)
    log_loss(y, sigmoid(X @ w + b)).backward()  # autograd's answer...
    error = sigmoid(X @ w + b).detach() - y  # ...and the formula: predicted minus truth
    assert torch.allclose(w.grad, X.T @ error / n) and torch.isclose(b.grad, error.mean())
    print("gradient:", w.grad.round(decimals=4).tolist(), round(b.grad.item(), 4))
