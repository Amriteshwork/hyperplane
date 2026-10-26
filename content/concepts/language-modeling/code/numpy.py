"""Language modelling in NumPy: a bigram model, cross-entropy, perplexity, and sampling with temperature."""

import random

import numpy as np

# @a:reuse
from log_loss_numpy import log_loss
from softmax_numpy import softmax
from tokenization_numpy import build_vocab, encode

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
    """P[a, b] = P(next is b | current is a): count every pair, plus `smoothing` so nothing is 0."""
    counts = np.full((V, V), smoothing)
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


# @a:cross-entropy
def cross_entropy(probs, targets):
    """Average −log of the probability given to the token that really came next."""
    return -np.mean(np.log(probs[np.arange(len(targets)), targets]))


# @a:sample
def sample(P, start, n, rng, temperature=1.0):
    """Generate `n` tokens: look up the next-token distribution, sharpen or flatten it, draw."""
    ids = [start]
    for _ in range(n):
        p = softmax(np.log(P[ids[-1]])[None] / temperature)[0]
        ids.append(int(rng.choice(len(p), p=p)))
    return ids


# @a:demo
if __name__ == "__main__":
    # @a:fit
    text = make_corpus(1000)
    vocab = build_vocab(text)
    ids = encode(text, vocab)
    split = int(0.9 * len(ids))
    train, val = ids[:split], ids[split:]  # judge the model on text it hasn't counted
    P = fit_bigram(train, len(vocab))

    # @a:scores
    loss = cross_entropy(P[val[:-1]], val[1:])
    print(f"{len(vocab)} characters; uniform guessing: {np.log(len(vocab)):.2f} nats, bigram: {loss:.2f} "
          f"(perplexity {np.exp(loss):.1f})")
    assert loss < np.log(len(vocab)) / 2  # far better than guessing...
    assert np.isclose(cross_entropy(np.array([[0.3, 0.7]]), [1]), log_loss(1, 0.7))  # 2 tokens: log-loss

    # @a:blind-spot
    itos = {i: ch for ch, i in vocab.items()}
    for name in ("zoe", "max"):
        after_so = P[vocab[" "]][vocab[name[0]]]  # P(first letter of the name | previous char ' ')
        print(f"after 'so ', P({name[0]!r}) = {after_so:.3f}, even when the sentence began with {name!r}")
    assert P[vocab[" "]].max() < 0.5  # ...but it can't look back to see whose sentence it is

    # @a:temperature
    rng = np.random.default_rng(0)
    for temperature in (1.0, 0.001):
        out = sample(P, vocab["z"], 40, rng, temperature)
        print(f"temperature {temperature}: {''.join(itos[i] for i in out)!r}")
    assert all(P[a, b] == P[a].max() for a, b in zip(out, out[1:]))  # temperature → 0: argmax
    assert P[vocab["e"], vocab["s"]] == P[vocab["e"], vocab["a"]]  # (a tie: likes/eats, a coin flip)
