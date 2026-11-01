"""Likelihood in PyTorch: maximise it with autograd, and land on the closed-form answer."""

import math

import torch


# @a:log-pdf
def log_normal_pdf(x, mu=0.0, variance=1.0):
    """The log of the bell curve's height: safe where the height itself would round to 0."""
    variance = torch.as_tensor(variance, dtype=torch.float64)
    return -0.5 * torch.log(2 * math.pi * variance) - (x - mu) ** 2 / (2 * variance)


# @a:log-likelihood
def log_likelihood(x, mu, variance):
    """How well N(mu, variance) explains every row of x, assuming the rows are independent."""
    return log_normal_pdf(x, mu, variance).sum()


# @a:fit
def fit_gaussian(x):
    """The maximum-likelihood mean and variance: just the sample's own."""
    return x.mean(), ((x - x.mean()) ** 2).mean()


# @a:demo
if __name__ == "__main__":
    torch.manual_seed(0)
    x = 10.0 + 2.0 * torch.randn(200, dtype=torch.float64)
    mu, variance = fit_gaussian(x)

    # @a:ascent
    guess_mu = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    guess_log_var = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)  # stays positive
    optimizer = torch.optim.Adam([guess_mu, guess_log_var], lr=0.05)
    for _ in range(3000):
        optimizer.zero_grad()
        loss = -log_likelihood(x, guess_mu, guess_log_var.exp()) / len(x)
        loss.backward()
        optimizer.step()
    print(f"gradient ascent: mu = {guess_mu.item():.3f}, variance = {guess_log_var.exp().item():.3f}")
    print(f"closed form:     mu = {mu:.3f}, variance = {variance:.3f}")
    assert abs(guess_mu - mu) < 1e-3 and abs(guess_log_var.exp() - variance) < 1e-3

    # @a:torch-distribution
    library = torch.distributions.Normal(mu, variance.sqrt()).log_prob(x).sum()
    assert abs(library - log_likelihood(x, mu, variance)) < 1e-9
