"""Residual connections in PyTorch: the same 30-layer network trains with skips and stalls without."""

import torch


# @a:block
class Block(torch.nn.Module):
    """One tanh layer, optionally wrapped as x + f(x)."""

    def __init__(self, width, skip):
        super().__init__()
        self.layer, self.skip = torch.nn.Linear(width, width), skip

    def forward(self, x):
        out = torch.tanh(self.layer(x))
        return x + out if self.skip else out


# @a:network
def deep_network(depth, skip, width=32):
    """Lift x to `width` numbers, run `depth` blocks, read one number out."""
    torch.manual_seed(0)  # identical starting weights with and without skips
    blocks = [Block(width, skip) for _ in range(depth)]
    return torch.nn.Sequential(torch.nn.Linear(1, width), *blocks, torch.nn.Linear(width, 1))


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, 128)[:, None]
    y = torch.sin(2 * x)

    # @a:first-layer
    for skip in (False, True):
        net = deep_network(30, skip)
        ((net(x) - y) ** 2).mean().backward()
        print(f"skip={skip!s:5}: gradient on the first block {net[1].layer.weight.grad.norm():.1e}, "
              f"last block {net[30].layer.weight.grad.norm():.1e}")

    # @a:train
    losses = {}
    for skip in (False, True):
        net = deep_network(30, skip)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        for step in range(200):
            loss = ((net(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses[skip] = loss.item()
    print(f"after 200 steps: plain MSE {losses[False]:.3f}, residual MSE {losses[True]:.3f}")
    assert losses[True] < 0.05 < 0.3 < losses[False]  # predicting 0 everywhere scores ≈ 0.5
