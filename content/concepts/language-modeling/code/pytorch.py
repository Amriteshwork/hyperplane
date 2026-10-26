"""Language modelling in PyTorch: a bigram model as a network, trained with cross-entropy, then sampled."""

import math
import random

import torch
import torch.nn.functional as F

# @a:reuse
from tokenization_pytorch import build_vocab, decode, encode

# @a:corpus
NAMES = ["ann", "bob", "eve", "max", "zoe", "ian", "kim", "raj"]
FOODS = ["figs", "plums", "rice", "nuts", "soup", "kale", "eggs", "jam"]


def make_corpus(n, seed=0):
    """`n` sentences like 'zoe likes jam, so zoe eats jam. ': the second half repeats the first."""
    pick = random.Random(seed)
    return "".join(
        f"{a} likes {f}, so {a} eats {f}. "
        for a, f in ((pick.choice(NAMES), pick.choice(FOODS)) for _ in range(n))
    )


# @a:bigram
def fit_bigram(ids, V, smoothing=1.0):
    """log P(next is b | current is a), from pair counts plus `smoothing`; shape (V, V)."""
    counts = torch.full((V, V), float(smoothing))
    counts.index_put_((ids[:-1], ids[1:]), torch.ones(len(ids) - 1), accumulate=True)
    return torch.log(counts / counts.sum(dim=1, keepdim=True))


# @a:generate
@torch.no_grad()
def generate(model, ids, n, temperature=1.0, generator=None):
    """Append `n` tokens, each drawn from the model's prediction at the last position."""
    for _ in range(n):
        logits = model(ids)[-1] / temperature  # (V,) scores for the next token
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
        ids = torch.cat([ids, next_id])
    return ids


# @a:demo
if __name__ == "__main__":
    # @a:data
    text = make_corpus(1000)
    vocab = build_vocab(text)
    V = len(vocab)
    ids = encode(text, vocab)
    split = int(0.9 * len(ids))
    train, val = ids[:split], ids[split:]

    # @a:model
    torch.manual_seed(0)
    bigram = torch.nn.Embedding(V, V)  # row a = the scores (logits) for whatever follows a
    optimizer = torch.optim.Adam(bigram.parameters(), lr=0.1)
    for step in range(300):
        loss = F.cross_entropy(bigram(train[:-1]), train[1:])  # softmax + log-loss in one call
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # @a:counts
    counted = -fit_bigram(train, V, smoothing=0)[train[:-1], train[1:]].mean().item()  # raw counts
    with torch.no_grad():
        val_loss = F.cross_entropy(bigram(val[:-1]), val[1:]).item()
    print(f"trained {loss.item():.3f} vs counted {counted:.3f}; validation {val_loss:.3f}, "
          f"perplexity {math.exp(val_loss):.1f} (uniform guessing: {V})")
    assert abs(loss.item() - counted) < 0.01  # gradient descent rediscovers the counts

    # @a:sampling
    rng = torch.Generator().manual_seed(0)
    start = encode("z", vocab)
    for temperature in (0.5, 1.0, 2.0):
        out = generate(bigram, start, 40, temperature, rng)
        print(f"temperature {temperature}: {decode(out, vocab)!r}")
