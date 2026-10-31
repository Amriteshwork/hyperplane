"""Warmup in PyTorch: the same tiny GPT trained at a learning rate that breaks it, then rescued one trick at a time."""

import math

import torch
import torch.nn.functional as F

# @a:reuse
from language_modeling_pytorch import make_corpus
from tokenization_pytorch import build_vocab, encode, windows
from transformer_pytorch import TinyGPT


# @a:schedule
def learning_rate(step, peak, warmup, total):
    """A linear climb to `peak`, then a half cosine down to 0. `warmup=0` means a constant rate."""
    if not warmup:
        return peak
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


# @a:run
def fit(x, y, x_val, y_val, peak, warmup=0, clip=None, tie=False, steps=200):
    """One training run. Everything shared is seeded, so runs differ only by the tricks switched on."""
    torch.manual_seed(0)
    V = int(y.max()) + 1
    model = TinyGPT(V)
    if tie:
        model.unembed.weight = model.embed.weight  # one table, used in both directions
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, peak, warmup, steps)
        batch = torch.randint(len(x), (32,))
        loss = F.cross_entropy(model(x[batch]).reshape(-1, V), y[batch].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    with torch.no_grad():
        return F.cross_entropy(model(x_val).reshape(-1, V), y_val.reshape(-1)).item(), model


# @a:demo
if __name__ == "__main__":
    # @a:data
    text = make_corpus(1000)
    vocab = build_vocab(text)
    ids = encode(text, vocab)
    split = int(0.9 * len(ids))
    x, y = windows(ids[:split], 24)
    x_val, y_val = windows(ids[split:], 24)
    x_val, y_val = x_val[::24], y_val[::24]
    args = (x, y, x_val, y_val)

    # @a:rescue
    calm, untied_model = fit(*args, peak=1e-2)
    wild = fit(*args, peak=0.1)[0]
    clipped = fit(*args, peak=0.1, clip=1.0)[0]
    warmed = fit(*args, peak=0.1, warmup=40)[0]
    print(f"lr 0.01, constant:   validation loss {calm:.3f}")
    print(f"lr 0.1,  constant:   validation loss {wild:.3f}   too fast, and it never recovers")
    print(f"lr 0.1,  clipping:   validation loss {clipped:.3f}   clipping alone doesn't save it")
    print(f"lr 0.1,  warmup:     validation loss {warmed:.3f}   the same peak, reached gradually")
    assert wild > 3 * warmed and warmed < calm * 1.2

    # @a:tied
    tied, tied_model = fit(*args, peak=1e-2, tie=True)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    print(f"weight tying: {count(untied_model):,} → {count(tied_model):,} parameters, "
          f"validation loss {calm:.3f} → {tied:.3f}")
    assert count(tied_model) < count(untied_model)
