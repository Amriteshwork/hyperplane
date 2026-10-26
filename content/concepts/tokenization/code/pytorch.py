"""Tokenization in PyTorch: text becomes a tensor of ids, cut into next-token training windows."""

import torch


# @a:vocab
def build_vocab(text):
    """Each distinct character gets an id; sorted, so the same text always gives the same ids."""
    return {ch: i for i, ch in enumerate(sorted(set(text)))}


# @a:encode
def encode(text, vocab):
    """Text → 1-D tensor of token ids (`long`, the dtype embedding layers expect)."""
    return torch.tensor([vocab[ch] for ch in text], dtype=torch.long)


# @a:decode
def decode(ids, vocab):
    """Token ids → text, by reading the vocabulary backwards."""
    itos = {i: ch for ch, i in vocab.items()}
    return "".join(itos[int(i)] for i in ids)


# @a:windows
def windows(ids, length):
    """Every run of `length` tokens (x), paired with the same run shifted one ahead (y)."""
    chunks = ids.unfold(0, length + 1, 1)  # sliding windows, one token longer than needed
    return chunks[:, :-1], chunks[:, 1:]


# @a:demo
if __name__ == "__main__":
    # @a:round-trip
    text = "the cat sat on the mat. the cat ate the hat."
    vocab = build_vocab(text)
    ids = encode(text, vocab)
    print(f"{len(vocab)} characters in the vocabulary; first ids: {ids[:10].tolist()}")
    assert decode(ids, vocab) == text  # lossless: nothing is thrown away

    # @a:targets
    x, y = windows(ids, length=8)
    print(f"{len(x)} windows; the first one asks the model to predict, at every position:")
    for t in range(4):
        print(f"  after {decode(x[0, : t + 1], vocab)!r:>8} comes {decode(y[0, t : t + 1], vocab)!r}")
    assert x.shape == y.shape == (len(ids) - 8, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])  # the target is the input, one step ahead
