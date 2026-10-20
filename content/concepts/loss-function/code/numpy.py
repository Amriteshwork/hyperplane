"""Loss function in NumPy: one number that says how wrong a set of predictions is."""

import numpy as np

# @a:data
rng = np.random.default_rng(0)
y_true = rng.normal(size=100)  # 100 random targets
good = y_true + rng.normal(scale=0.1, size=100)  # predictions close to the truth
bad = y_true + rng.normal(scale=1.0, size=100)  # predictions far from it


# @a:losses
def squared_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def absolute_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# @a:compare
for name, loss in [("squared", squared_loss), ("absolute", absolute_loss)]:
    print(f"{name:>8} loss: good {loss(y_true, good):.4f}   bad {loss(y_true, bad):.4f}")
    assert loss(y_true, good) < loss(y_true, bad)  # lower loss = better predictions

# @a:outlier
# One big miss: squared loss blows up, absolute loss barely moves. The choice of loss
# decides what the model cares about.
outlier = good.copy()
outlier[0] += 10
squared_growth = squared_loss(y_true, outlier) / squared_loss(y_true, good)
absolute_growth = absolute_loss(y_true, outlier) / absolute_loss(y_true, good)
print(f"one outlier: squared loss x{squared_growth:.0f}, absolute loss x{absolute_growth:.1f}")
assert squared_growth > absolute_growth
