"""The curse of dimensionality in PyTorch: the same concentration, measured with `torch.cdist`."""

import torch


# @a:contrast
def relative_contrast(X, queries):
    """Mean of (farthest − nearest) / nearest, over the queries."""
    d = torch.cdist(queries, X)
    return ((d.max(dim=1).values - d.min(dim=1).values) / d.min(dim=1).values).mean().item()


# @a:demo
if __name__ == "__main__":
    torch.manual_seed(0)

    # @a:concentration
    contrast = {}
    for d in (2, 10, 100, 1000):
        contrast[d] = relative_contrast(torch.rand(1000, d), torch.rand(20, d))
        print(f"d = {d:4d}: farthest is {1 + contrast[d]:7.2f} × as far as nearest")
    assert contrast[2] > 20 * contrast[100] > 20 * contrast[1000]
    assert contrast[1000] < 0.3

    # @a:spread
    spread = {}
    for d in (2, 1000):
        distances = torch.cdist(torch.rand(1, d), torch.rand(1000, d))
        spread[d] = (distances.std() / distances.mean()).item()
        print(f"d = {d:4d}: spread of distances is {100 * spread[d]:.1f}% of their mean")
    assert spread[1000] < 0.1 * spread[2]  # every row sits at nearly the same distance
