"""Loss function in PyTorch: one number that says how wrong a set of predictions is."""

import torch
import torch.nn.functional as F

# @a:data
torch.manual_seed(0)
y_true = torch.randn(100)  # 100 random targets
good = y_true + 0.1 * torch.randn(100)  # predictions close to the truth
bad = y_true + 1.0 * torch.randn(100)  # predictions far from it


# @a:losses
def squared_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def absolute_loss(y_true, y_pred):
    return (y_true - y_pred).abs().mean()


# @a:compare
for name, loss in [("squared", squared_loss), ("absolute", absolute_loss)]:
    print(f"{name:>8} loss: good {loss(y_true, good):.4f}   bad {loss(y_true, bad):.4f}")
    assert loss(y_true, good) < loss(y_true, bad)  # lower loss = better predictions

# @a:builtin
# PyTorch ships both: mse_loss and l1_loss compute exactly these numbers.
assert torch.isclose(squared_loss(y_true, good), F.mse_loss(good, y_true))
assert torch.isclose(absolute_loss(y_true, good), F.l1_loss(good, y_true))

# @a:outlier
# One big miss: squared loss blows up, absolute loss barely moves. The choice of loss
# decides what the model cares about.
outlier = good.clone()
outlier[0] += 10
squared_growth = squared_loss(y_true, outlier) / squared_loss(y_true, good)
absolute_growth = absolute_loss(y_true, outlier) / absolute_loss(y_true, good)
print(f"one outlier: squared loss x{squared_growth:.0f}, absolute loss x{absolute_growth:.1f}")
assert squared_growth > absolute_growth
