"""Word2vec in NumPy: skip-gram with negative sampling, trained by hand until words that act alike sit together."""

import random

import numpy as np

# @a:reuse
from embeddings_numpy import cosine
from sigmoid_numpy import sigmoid
from tokenization_numpy import build_vocab

# @a:corpus
GROUPS = {
    "animal": ["cat", "dog", "fox", "owl"],
    "food": ["rice", "soup", "figs", "jam"],
    "vehicle": ["car", "bus", "tram", "van"],
}
PATTERNS = {
    "animal": ["the {} sleeps", "a {} runs fast", "the {} hunts at night", "my {} chases birds"],
    "food": ["we cook {}", "hot {} tastes good", "i eat {} at lunch", "fresh {} on my plate"],
    "vehicle": ["the {} stops here", "a {} needs fuel", "we drive the {}", "my {} has wheels"],
}


def make_sentences(n, seed=0):
    """`n` tokenised sentences. Nothing says which words are related, only what they appear next to."""
    pick = random.Random(seed)
    sentences = []
    for _ in range(n):
        group = pick.choice(list(GROUPS))
        sentences.append(pick.choice(PATTERNS[group]).format(pick.choice(GROUPS[group])).split())
    return sentences


# @a:pairs
def skipgram_pairs(sentences, vocab, window=2):
    """Every (word, a word within `window` of it). This is the whole dataset word2vec sees."""
    pairs = []
    for words in sentences:
        ids = [vocab[w] for w in words]
        for i, center in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((center, ids[j]))
    return np.array(pairs)


# @a:loss
def sgns_loss(u, v, negatives):
    """Real neighbour v: push u·v up. Random words: push u·n down. Returns the loss and its gradients."""
    pos, neg = sigmoid(u @ v), sigmoid(negatives @ u)
    loss = -np.log(pos) - np.log(1 - neg).sum()
    return loss, -(1 - pos) * v + neg @ negatives, -(1 - pos) * u, neg[:, None] * u


# @a:train
def train(pairs, V, d=16, k=4, lr=0.05, epochs=4, seed=0):
    """Plain SGD over the pairs. E holds each word as a centre; C holds it as a context."""
    rng = np.random.default_rng(seed)
    E, C = rng.normal(size=(V, d)) * 0.1, rng.normal(size=(V, d)) * 0.1
    for _ in range(epochs):
        total = 0.0
        for center, context in pairs[rng.permutation(len(pairs))]:
            drawn = rng.integers(V, size=k)  # random "fake neighbours"
            loss, du, dv, dn = sgns_loss(E[center], C[context], C[drawn])
            E[center] -= lr * du
            C[context] -= lr * dv
            np.subtract.at(C, drawn, lr * dn)
            total += loss
        print(f"  mean loss {total / len(pairs):.3f}")
    return E


# @a:neighbours
def neighbours(word, E, vocab, n=3):
    """The `n` words whose vectors point most nearly the same way."""
    names = list(vocab)
    sims = [(cosine(E[vocab[word]], E[i]), names[i]) for i in range(len(names)) if names[i] != word]
    return [w for _, w in sorted(sims, reverse=True)[:n]]


# @a:demo
if __name__ == "__main__":
    # @a:data
    sentences = make_sentences(1500)
    vocab = build_vocab(sorted({w for s in sentences for w in s}))
    pairs = skipgram_pairs(sentences, vocab)
    print(f"{len(sentences)} sentences → {len(pairs)} (word, neighbour) pairs, vocabulary {len(vocab)}")
    print(f"e.g. {sentences[0]}")

    # @a:gradient-check
    rng = np.random.default_rng(1)
    u, v, n = rng.normal(size=4), rng.normal(size=4), rng.normal(size=(2, 4))
    _, du, _, _ = sgns_loss(u, v, n)
    eps = 1e-6
    numeric = [(sgns_loss(u + eps * e, v, n)[0] - sgns_loss(u - eps * e, v, n)[0]) / (2 * eps) for e in np.eye(4)]
    assert np.allclose(du, numeric, atol=1e-6)  # the hand-derived gradient is right

    # @a:fit
    E = train(pairs, len(vocab))

    # @a:result
    for word in ("cat", "soup", "tram"):
        print(f"  nearest to {word}: {neighbours(word, E, vocab)}")
    for group, words in GROUPS.items():
        others = [w for g, ws in GROUPS.items() if g != group for w in ws]
        same = np.mean([cosine(E[vocab[a]], E[vocab[b]]) for a in words for b in words if a != b])
        different = np.mean([cosine(E[vocab[a]], E[vocab[b]]) for a in words for b in others])
        print(f"  {group}: same group {same:.2f}, other groups {different:.2f}")
        assert same > different + 0.3  # nobody told it the groups
