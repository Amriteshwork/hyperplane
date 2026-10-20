"""Bias and variance in PyTorch: refit on many fresh samples and see how predictions spread."""

import torch

# @a:setup
torch.manual_seed(0)
x_test = torch.linspace(-0.9, 0.9, 50, dtype=torch.float64)
truth = torch.sin(3 * x_test)  # the function every training sample comes from


# @a:refit
def predictions(degree, samples=300):
    preds = []
    for _ in range(samples):  # a fresh, noisy training set each time
        x = torch.rand(40, dtype=torch.float64) * 2 - 1
        y = torch.sin(3 * x) + 0.3 * torch.randn(40, dtype=torch.float64)
        A = torch.linalg.vander(x, N=degree + 1)
        beta = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
        preds.append(torch.linalg.vander(x_test, N=degree + 1) @ beta)
    return torch.stack(preds)  # shape (samples, points)


# @a:decompose
def bias_variance(degree):
    preds = predictions(degree)
    bias2 = ((preds.mean(dim=0) - truth) ** 2).mean().item()  # how wrong on average
    variance = preds.var(dim=0, correction=0).mean().item()  # swing between samples
    print(f"degree {degree}: bias² {bias2:.4f}  variance {variance:.4f}")
    return bias2, variance


simple, balanced, flexible = (bias_variance(degree) for degree in (1, 3, 8))

# @a:checks
assert simple[0] > 10 * balanced[0]  # the line is wrong the same way every time
assert flexible[1] > 3 * balanced[1]  # the wiggly fit changes with every sample
assert sum(balanced) < min(sum(simple), sum(flexible))  # the sweet spot in between
