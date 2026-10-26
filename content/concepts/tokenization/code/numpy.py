"""Tokenization in NumPy: text becomes integers and back, plus a few byte-pair-encoding merges."""

from collections import Counter

import numpy as np


# @a:vocab
def build_vocab(text):
    """Each distinct character gets an id; sorted, so the same text always gives the same ids."""
    return {ch: i for i, ch in enumerate(sorted(set(text)))}


# @a:encode
def encode(text, vocab):
    """Text → array of token ids."""
    return np.array([vocab[ch] for ch in text])


# @a:decode
def decode(ids, vocab):
    """Token ids → text, by reading the vocabulary backwards."""
    itos = {i: ch for ch, i in vocab.items()}
    return "".join(itos[int(i)] for i in ids)


# @a:bpe
def most_common_pair(ids):
    """The pair of neighbouring tokens that appears most often (first seen wins a tie)."""
    return Counter(zip(ids[:-1], ids[1:])).most_common(1)[0][0]


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` with the single token `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# @a:demo
if __name__ == "__main__":
    # @a:round-trip
    text = "the cat sat on the mat. the cat ate the hat."
    vocab = build_vocab(text)
    ids = encode(text, vocab)
    print(f"{len(vocab)} characters in the vocabulary; first ids: {ids[:10]}")
    assert decode(ids, vocab) == text  # lossless: nothing is thrown away

    # @a:merges
    tokens, merges = [int(i) for i in ids], {}
    pieces = {i: ch for ch, i in vocab.items()}  # what each token spells, for printing
    for new_id in range(len(vocab), len(vocab) + 5):
        a, b = most_common_pair(tokens)
        tokens, merges[new_id] = merge(tokens, (a, b), new_id), (a, b)
        pieces[new_id] = pieces[a] + pieces[b]
        print(f"merge {pieces[a]!r} + {pieces[b]!r} -> {pieces[new_id]!r}: {len(tokens)} tokens")
    assert len(tokens) < len(ids)  # same text, fewer tokens

    # @a:unmerge
    for new_id, pair in reversed(merges.items()):  # undo the merges, newest first
        tokens = [t for tok in tokens for t in (pair if tok == new_id else (tok,))]
    assert tokens == list(ids)  # merging is lossless too
