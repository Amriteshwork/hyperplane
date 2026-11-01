"""Bayes' theorem in PyTorch: the same update, and a check against counting simulated people."""

import torch


# @a:posterior
def posterior(prior, likelihood):
    """P(class | evidence) from P(class) and P(evidence | class); the last axis is the classes."""
    joint = prior * likelihood
    return joint / joint.sum(dim=-1, keepdim=True)


# @a:demo
if __name__ == "__main__":
    torch.manual_seed(0)

    # @a:test
    prior = torch.tensor([0.99, 0.01], dtype=torch.float64)  # healthy, sick
    positive = torch.tensor([0.05, 0.90], dtype=torch.float64)
    sick_given_positive = posterior(prior, positive)[1]
    print(f"P(sick | positive) = {sick_given_positive:.3f}, though P(positive | sick) = {positive[1]:.2f}")
    assert abs(sick_given_positive - 0.009 / (0.009 + 0.0495)) < 1e-12

    # @a:simulate
    n = 2_000_000
    sick = torch.rand(n) < prior[1]
    tested_positive = torch.rand(n, dtype=torch.float64) < positive[sick.long()]
    counted = sick[tested_positive].double().mean()
    print(f"share of simulated positives who are sick: {counted:.3f}")
    assert abs(counted - sick_given_positive) < 0.003

    # @a:update
    twice = posterior(posterior(prior, positive), positive)[1]
    print(f"after a second independent positive: {twice:.3f}")
    assert abs(twice - posterior(prior, positive**2)[1]) < 1e-12  # the same as both at once
