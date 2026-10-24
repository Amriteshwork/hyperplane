"""Adam in PyTorch: torch.optim.Adam, checked against the update rule by hand."""

import torch

# @a:by-hand
torch.manual_seed(0)
w = torch.randn(3, requires_grad=True)
optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
m, v, expected = torch.zeros(3), torch.zeros(3), w.detach().clone()
for t in range(1, 6):
    loss = (torch.tensor([100.0, 1.0, 0.01]) * w**2).sum()  # very different curvatures
    optimizer.zero_grad()
    loss.backward()
    g = w.grad.detach()
    m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g**2
    expected -= 0.01 * (m / (1 - 0.9**t)) / ((v / (1 - 0.999**t)).sqrt() + 1e-8)
    optimizer.step()
assert torch.allclose(w.detach(), expected, atol=1e-6)
print("torch.optim.Adam follows the update rule exactly:", w.detach().numpy().round(4))

# @a:network
x = torch.rand(200, 1) * 6 - 3
y = torch.sin(2 * x) + 0.1 * torch.randn(200, 1)
network = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
for step in range(1001):
    loss = ((network(x) - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"step {step:4d}: MSE {loss.item():.4f}")
assert loss.item() < 0.03
