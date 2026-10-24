"""Neurons and layers in PyTorch: nn.Linear is a layer's weights and biases; stack them."""

import torch
from torch import nn

# @a:shapes
torch.manual_seed(0)
X = torch.randn(5, 3)  # 5 rows, 3 features
network = nn.Sequential(
    nn.Linear(3, 4), nn.Sigmoid(),  # 3 inputs → 4 hidden neurons
    nn.Linear(4, 1), nn.Sigmoid(),  # 4 hidden → 1 output
)
H = network[1](network[0](X))
print("hidden layer:", tuple(H.shape), "| output:", tuple(network(X).shape))
first = network[0]
assert torch.allclose(H, torch.sigmoid(X @ first.weight.T + first.bias))  # weight is (out, in)
assert sum(p.numel() for p in network.parameters()) == 21

# @a:xor-network
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0, 1, 1, 0])
xor = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid())
with torch.no_grad():  # set the weights by hand: hidden OR and AND, output "OR and not AND"
    xor[0].weight.copy_(torch.tensor([[20.0, 20.0], [20.0, 20.0]]))
    xor[0].bias.copy_(torch.tensor([-10.0, -30.0]))
    xor[2].weight.copy_(torch.tensor([[20.0, -20.0]]))
    xor[2].bias.copy_(torch.tensor([-10.0]))
    out = xor(X)[:, 0]
print("XOR:", out.numpy().round(2))
assert torch.equal((out > 0.5).long(), y)
