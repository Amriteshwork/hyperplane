"""PyTorch tensors: the same array ideas as NumPy, plus GPUs and automatic gradients."""

import torch

# @a:arrays
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
X = torch.arange(12, dtype=torch.float32).reshape(4, 3)  # 4 rows, 3 columns
print("x.shape =", tuple(x.shape), "  X.shape =", tuple(X.shape))

# @a:vectorised
doubled = x * 2  # every element at once, no loop
looped = torch.tensor([value * 2 for value in x.tolist()])  # the same, the slow way
assert torch.equal(doubled, looped)

# @a:axis
column_means = X.mean(dim=0)  # `dim` is PyTorch's name for NumPy's `axis`
row_sums = X.sum(dim=1)
print("column means:", column_means.tolist(), "  row sums:", row_sums.tolist())
assert column_means.shape == (3,) and row_sums.shape == (4,)

# @a:broadcasting
centred = X - column_means  # (4, 3) − (3,): the same broadcasting rules as NumPy
assert torch.allclose(centred.mean(dim=0), torch.zeros(3))

# @a:mask
big = X[X[:, 0] > 4]  # keep rows whose first feature exceeds 4
print("rows with first feature > 4:\n", big)
assert big.shape == (2, 3)

# @a:matmul
w = torch.tensor([0.5, -1.0, 2.0])
predictions = X @ w  # one weighted sum per row: (4, 3) @ (3,) → (4,)
assert torch.allclose(predictions, torch.stack([(row * w).sum() for row in X]))
