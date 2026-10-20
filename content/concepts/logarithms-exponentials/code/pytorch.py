"""Logarithms and exponentials in PyTorch: products become sums, and how to add up huge exponentials."""

import torch


# @a:logsumexp
def logsumexp(z, dim=None):
    """log(Σ e^z) without overflow: factor out the largest term first."""
    if dim is None:
        z, dim = z.reshape(-1), 0
    top = z.max(dim=dim, keepdim=True).values
    return (top + torch.log(torch.exp(z - top).sum(dim=dim, keepdim=True))).squeeze(dim)


# @a:demo
if __name__ == "__main__":
    f64 = torch.float64

    # @a:inverse
    x = torch.linspace(-3, 3, 7, dtype=f64)
    assert torch.allclose(torch.log(torch.exp(x)), x)  # log undoes exp...
    assert torch.allclose(torch.exp(torch.log(x[x > 0])), x[x > 0])  # ...and exp undoes log

    # @a:products
    torch.manual_seed(0)
    p = 0.1 + 0.9 * torch.rand(2000, dtype=f64)  # 2000 probabilities, e.g. one per training row
    print("product of probabilities:", torch.prod(p).item())  # underflows to exactly 0.0
    print("sum of their logs:       ", torch.log(p).sum().item())  # perfectly representable
    assert torch.prod(p) == 0.0 and torch.isfinite(torch.log(p).sum())
    assert torch.isclose(torch.log(torch.prod(p[:20])), torch.log(p[:20]).sum())  # log ab = log a + log b

    # @a:derivative
    t = torch.tensor([1.0, 2.0], dtype=f64, requires_grad=True)
    torch.exp(t[0]).backward()
    torch.log(t[1]).backward()
    assert torch.isclose(t.grad[0], torch.exp(t[0].detach()))  # (eˣ)' = eˣ
    assert torch.isclose(t.grad[1], torch.tensor(0.5, dtype=f64))  # (log x)' = 1/x

    # @a:stable
    z = torch.tensor([1000.0, 1001.0, 1002.0], dtype=f64)
    naive = torch.log(torch.exp(z).sum())  # e¹⁰⁰⁰ overflows to inf
    print(f"naive: {naive.item()}, logsumexp: {logsumexp(z).item():.4f}")
    assert torch.isinf(naive)
    assert torch.isclose(logsumexp(z), torch.logsumexp(z, dim=0))  # PyTorch's own version agrees
