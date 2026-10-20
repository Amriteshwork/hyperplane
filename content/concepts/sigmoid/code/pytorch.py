"""The sigmoid in PyTorch: squash any score into a probability, and undo it with the log-odds."""

import torch


# @a:sigmoid
def sigmoid(z):
    """1 / (1 + e^−z), computed so that large |z| never overflows."""
    return torch.exp(-torch.logaddexp(torch.zeros_like(z), -z))


# @a:demo
if __name__ == "__main__":
    f64 = torch.float64

    # @a:shape
    z = torch.tensor([-10.0, -2.0, 0.0, 2.0, 10.0], dtype=f64)
    print("z:         ", z.tolist())
    print("sigmoid(z):", sigmoid(z).round(decimals=4).tolist())
    assert sigmoid(torch.tensor(0.0)) == 0.5  # a score of 0 means a coin flip
    assert torch.all((0 < sigmoid(z)) & (sigmoid(z) < 1))  # always a valid probability
    assert torch.allclose(sigmoid(-z), 1 - sigmoid(z))  # symmetric around (0, 0.5)
    assert torch.allclose(sigmoid(z), torch.sigmoid(z))  # PyTorch's built-in agrees

    # @a:log-odds
    p = sigmoid(z)
    assert torch.allclose(torch.log(p / (1 - p)), z)  # the logit: sigmoid's inverse

    # @a:derivative
    zg = z.clone().requires_grad_(True)
    sigmoid(zg).sum().backward()  # each output depends on its own input only
    assert torch.allclose(zg.grad, p * (1 - p))  # σ' = σ(1 − σ)
    assert torch.isclose(zg.grad.max(), torch.tensor(0.25, dtype=f64))  # steepest at 0: slope ¼

    # @a:stable
    big = torch.tensor([-800.0, 800.0], dtype=f64)
    assert torch.allclose(sigmoid(big), torch.tensor([0.0, 1.0], dtype=f64))
    print("sigmoid(±800):", sigmoid(big).tolist())
