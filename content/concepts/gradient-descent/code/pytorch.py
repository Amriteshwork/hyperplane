"""Gradient descent in PyTorch: autograd computes the gradient, we take the steps."""

import torch

# @a:data
torch.manual_seed(0)
n = 500
X = torch.randn(n, 2)  # already standardised: mean 0, variance 1
y = 4.0 + X @ torch.tensor([2.0, -3.0]) + 0.5 * torch.randn(n)

# @a:init
w = torch.zeros(2, requires_grad=True)  # track operations on w and b to differentiate them
b = torch.zeros(1, requires_grad=True)
learning_rate = 0.05  # η: too big and the loss explodes, too small and it crawls
# @a:loop
for step in range(300):
    loss = ((X @ w + b - y) ** 2).mean()  # MSE
    # @a:backward
    loss.backward()  # fills w.grad and b.grad with ∂loss/∂w and ∂loss/∂b
    # @a:update
    with torch.no_grad():  # the update itself is not part of the model
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    w.grad.zero_()  # gradients accumulate; reset them for the next step
    b.grad.zero_()
    if step % 30 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")
# torch.optim.SGD([w, b], lr=learning_rate) does the same update in one `optimizer.step()`.

# @a:compare
# It lands where the normal equation jumps to in one step.
X1 = torch.cat([torch.ones(n, 1), X], dim=1)
exact = torch.linalg.lstsq(X1, y.unsqueeze(1)).solution.squeeze(1)
learned = torch.cat([b, w]).detach()
print("gradient descent:", learned.numpy().round(4))
print("normal equation: ", exact.numpy().round(4))
assert torch.allclose(learned, exact, atol=1e-3)
