"""Chat fine-tuning in NumPy: a conversation is one sequence, and the labels teach only the assistant's turns."""

import numpy as np

# @a:reuse
from language_modeling_numpy import cross_entropy
from softmax_numpy import softmax
from tokenization_numpy import build_vocab, encode

IGNORE = -100


# @a:template
def render_chat(turns):
    """[('user', 'hi'), ('assistant', 'hello')] → one string, with markers saying who speaks."""
    return "".join(f"<{role[0]}>{text}" + ("|" if role == "assistant" else "") for role, text in turns)


# @a:labels
def chat_labels(turns, vocab):
    """Inputs and next-token targets, with IGNORE wherever the next token isn't the assistant's."""
    ids, labels = [], []
    for role, text in turns:
        marker = encode(f"<{role[0]}>", vocab).tolist()
        body = encode(text + ("|" if role == "assistant" else ""), vocab).tolist()
        ids += marker + body
        labels += [IGNORE] * len(marker) + (body if role == "assistant" else [IGNORE] * len(body))
    return np.array(ids[:-1]), np.array(labels[1:])


# @a:masked-loss
def masked_cross_entropy(logits, labels):
    """Cross-entropy over the positions that have a real target; everything else costs nothing."""
    keep = labels != IGNORE
    return cross_entropy(softmax(logits[keep]), labels[keep])


# @a:demo
if __name__ == "__main__":
    # @a:render
    turns = [("user", "what does zoe eat?"), ("assistant", "jam."), ("user", "and raj?"), ("assistant", "figs.")]
    chat = render_chat(turns)
    print(chat)
    vocab = build_vocab(chat)
    assert chat == "<u>what does zoe eat?<a>jam.|<u>and raj?<a>figs.|"

    # @a:mask
    ids, labels = chat_labels(turns, vocab)
    assert len(ids) == len(labels) == len(encode(chat, vocab)) - 1
    itos = {i: c for c, i in vocab.items()}
    taught = "".join(itos[t] for t in labels if t != IGNORE)
    print(f"taught to predict: {taught!r}")
    assert taught == "jam.|figs.|"  # only the assistant's words, including when to stop
    print(f"{(labels == IGNORE).sum()} of {len(labels)} positions cost nothing")

    # @a:loss-check
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(len(ids), len(vocab)))
    base = masked_cross_entropy(logits, labels)
    user_turn = np.where(labels == IGNORE)[0]
    logits[user_turn] = rng.normal(size=(len(user_turn), len(vocab)))
    assert np.isclose(masked_cross_entropy(logits, labels), base)  # the user's turn can't change the loss
    print(f"loss {base:.3f}, whatever the model predicts during the user's turn")
