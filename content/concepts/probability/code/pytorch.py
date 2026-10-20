"""Probability in PyTorch: mean, variance and the normal distribution, checked by simulation."""

import math

import torch


# @a:moments
def mean(x):
    """The sample mean: the centre of the data."""
    return x.sum() / len(x)


def variance(x, ddof=0):
    """The average squared distance from the mean; ddof=1 gives the unbiased sample estimate."""
    return ((x - mean(x)) ** 2).sum() / (len(x) - ddof)


# @a:pdf
def normal_pdf(x, mu=0.0, sigma=1.0):
    """The bell curve: how dense the normal distribution N(μ, σ²) is at x."""
    return torch.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


# @a:demo
if __name__ == "__main__":
    torch.manual_seed(0)

    # @a:sample
    mu, sigma = 10.0, 2.0
    x = torch.normal(mu, sigma, size=(100_000,), dtype=torch.float64)
    print(f"sample mean {mean(x):.3f} (true {mu}),  sample std {variance(x).sqrt():.3f} (true {sigma})")
    assert abs(mean(x) - mu) < 0.05 and abs(variance(x).sqrt() - sigma) < 0.05
    assert torch.isclose(variance(x, ddof=1), x.var())  # torch.var is the ddof=1 estimate

    # @a:rule
    for k, expected in [(1, 0.683), (2, 0.954), (3, 0.997)]:
        share = ((x - mu).abs() < k * sigma).double().mean()
        print(f"within {k}σ: {share:.3f}")
        assert abs(share - expected) < 0.01

    # @a:density
    normal = torch.distributions.Normal(mu, sigma)  # PyTorch's built-in distribution
    grid = torch.linspace(mu - 6 * sigma, mu + 6 * sigma, 10_001, dtype=torch.float64)
    assert torch.allclose(normal_pdf(grid, mu, sigma), normal.log_prob(grid).exp())
    area = normal_pdf(grid, mu, sigma).sum() * (grid[1] - grid[0])
    assert abs(area - 1.0) < 1e-6  # a density's total area is 1

    # @a:large-numbers
    for n in [10, 1_000, 100_000]:
        rolls = torch.randint(1, 7, (n,)).double()
        print(f"mean of {n:>7,} dice rolls: {mean(rolls):.3f}  (true 3.5)")
    assert abs(mean(torch.randint(1, 7, (100_000,)).double()) - 3.5) < 0.03

    # @a:clt
    sums = torch.rand(20_000, 30, dtype=torch.float64).sum(dim=1)  # sums of 30 flat numbers
    z = (sums - 15) / math.sqrt(30 / 12)  # standardise with the sum's true mean and std
    assert abs((z.abs() < 1).double().mean() - 0.683) < 0.01  # behaves like a normal already
