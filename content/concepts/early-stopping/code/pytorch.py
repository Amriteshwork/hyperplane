"""Early stopping in PyTorch: copy the state_dict at the best validation loss, load it at the end."""

import copy

import torch

# @a:data
torch.manual_seed(0)


def noisy_sine(n):
    x = torch.rand(n, 1) * 6 - 3
    return x, torch.sin(x) + 0.3 * torch.randn(n, 1)


X_train, y_train = noisy_sine(30)
X_val, y_val = noisy_sine(200)

# @a:loop
network = torch.nn.Sequential(
    torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
)
optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
best, best_state, stale, patience = float("inf"), None, 0, 100
for epoch in range(1, 3001):
    loss = ((network(X_train) - y_train) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        val_loss = ((network(X_val) - y_val) ** 2).mean().item()
    # @a:check
    if val_loss < best:
        best, best_state, stale = val_loss, copy.deepcopy(network.state_dict()), 0
    else:
        stale += 1
    if stale >= patience:
        break
print(f"stopped at epoch {epoch}; best validation MSE {best:.3f} (last epoch {val_loss:.3f})")

# @a:restore
network.load_state_dict(best_state)  # back to the best epoch's weights
with torch.no_grad():
    assert abs(((network(X_val) - y_val) ** 2).mean().item() - best) < 1e-6
assert epoch < 3000 and best < 0.2
