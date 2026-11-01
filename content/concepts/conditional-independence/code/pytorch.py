"""Conditional independence in PyTorch: naive Bayes's sum over features, and its double counting."""

import torch

# @a:reuse
from likelihood_pytorch import log_normal_pdf


# @a:naive
def naive_posterior(x, priors, means, variances):
    """P(class | row), treating features as independent given the class: logs add, then softmax."""
    joint = priors.log() + log_normal_pdf(x[:, None, :], means, variances).sum(dim=2)
    return joint.softmax(dim=1)


# @a:demo
if __name__ == "__main__":
    torch.manual_seed(0)
    priors = torch.tensor([0.5, 0.5], dtype=torch.float64)
    centres = torch.tensor([-0.5, 0.5], dtype=torch.float64)

    # @a:double-count
    results = {}
    for copies in [1, 10]:
        y = torch.randint(0, 2, (100_000,))
        x = (centres[y] + torch.randn(100_000, dtype=torch.float64))[:, None].repeat(1, copies)
        means, variances = centres[:, None].repeat(1, copies), torch.ones(2, copies, dtype=torch.float64)
        probabilities = naive_posterior(x, priors, means, variances)
        results[copies] = (
            (probabilities.argmax(dim=1) == y).double().mean().item(),
            probabilities.max(dim=1).values.mean().item(),
        )
        print(f"{copies:2d} copies: accuracy {results[copies][0]:.3f}, average confidence {results[copies][1]:.3f}")
    assert abs(results[1][0] - results[1][1]) < 0.01  # one copy: the probabilities are right
    assert abs(results[1][0] - results[10][0]) < 0.01  # the labels barely change
    assert results[10][1] > results[10][0] + 0.1  # sure of itself, no better
