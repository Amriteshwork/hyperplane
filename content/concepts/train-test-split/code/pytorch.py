"""Train/test split in PyTorch: hold out rows to measure the model on data it never saw."""

import torch
from torch.utils.data import TensorDataset, random_split

# @a:data
generator = torch.Generator().manual_seed(42)  # fixed seed: same split, same numbers
n = 30  # small on purpose: flexible models overfit small datasets
x = torch.empty(n, dtype=torch.float64).uniform_(-1, 1, generator=generator)
y = torch.sin(3 * x) + 0.2 * torch.randn(n, dtype=torch.float64, generator=generator)


# @a:split
def train_test_split(n_rows, test_size, generator):
    order = torch.randperm(n_rows, generator=generator)  # shuffle so both parts look alike
    n_test = round(n_rows * test_size)
    return order[n_test:], order[:n_test]


# @a:check
train, test = train_test_split(n, test_size=0.3, generator=generator)
assert len(test) == 9
assert not set(train.tolist()) & set(test.tolist())  # no row is in both

# @a:random-split
# For training networks, random_split does the same on a Dataset.
train_set, test_set = random_split(TensorDataset(x, y), [0.7, 0.3], generator=generator)
assert (len(train_set), len(test_set)) == (21, 9)


# @a:features
def features(rows, degree):
    """Columns 1, x, x², …: polynomial regression is linear regression on these."""
    return torch.stack([x[rows] ** k for k in range(degree + 1)], dim=1)


def mse(beta, rows, degree):
    return ((features(rows, degree) @ beta - y[rows]) ** 2).mean().item()


# @a:fit
# Fit on the training rows only; score on both.
scores = {}
for degree in (1, 3, 12):
    beta = torch.linalg.lstsq(features(train, degree), y[train].unsqueeze(1)).solution.squeeze(1)
    scores[degree] = (mse(beta, train, degree), mse(beta, test, degree))
    print(f"degree {degree:2d}: train MSE {scores[degree][0]:.3f}   test MSE {scores[degree][1]:.3f}")

# @a:verdict
# More flexible models always fit the training rows better...
assert scores[1][0] > scores[3][0] > scores[12][0]
# ...but only the test rows reveal that degree 12 memorised noise.
assert scores[12][1] > scores[3][1]
