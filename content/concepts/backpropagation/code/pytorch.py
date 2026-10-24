"""Backpropagation in PyTorch: loss.backward() runs the chain rule over the graph your code built."""

import torch

# @a:data
torch.manual_seed(0)
x = torch.rand(200, 1) * 6 - 3
y = torch.sin(2 * x) + 0.1 * torch.randn(200, 1)

# @a:by-hand
W1, b1 = torch.randn(1, 16, requires_grad=True), torch.zeros(16, requires_grad=True)
W2, b2 = (torch.randn(16, 1) * 0.3).requires_grad_(), torch.zeros(1, requires_grad=True)
H = torch.tanh(x @ W1 + b1)
loss = ((H @ W2 + b2 - y) ** 2).mean()
loss.backward()  # fills .grad of every tensor that requires it
with torch.no_grad():
    delta2 = 2 * (H @ W2 + b2 - y) / len(y)  # the same recursion as the NumPy page
    delta1 = (delta2 @ W2.T) * (1 - H**2)
    assert torch.allclose(W2.grad, H.T @ delta2, atol=1e-6)
    assert torch.allclose(W1.grad, x.T @ delta1, atol=1e-6)
print("autograd's gradients match the hand-written backward pass")

# @a:train
network = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
for step in range(5001):
    loss = ((network(x) - y) ** 2).mean()
    optimizer.zero_grad()  # gradients add up across backward() calls unless cleared
    loss.backward()
    optimizer.step()
    if step % 1000 == 0:
        print(f"step {step:4d}: MSE {loss.item():.4f}")
assert loss.item() < 0.1  # a straight line can't get below about 0.45 here
