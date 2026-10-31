"""Chat fine-tuning in PyTorch: a base model that only continues text learns to answer, and to stop."""

import random

import torch
import torch.nn.functional as F

# @a:reuse
from language_modeling_pytorch import FOODS, NAMES, make_corpus
from tokenization_pytorch import build_vocab, decode, encode, windows
from transformer_pytorch import TinyGPT

CONTEXT = 56
STOP = "|"
#: Characters the template adds to the corpus's alphabet, so base and chat models share a vocabulary.
EXTRA = "<>?|uawhdt"


# @a:chat-data
def make_chat(pick, polite=False):
    """A fact, a question and an answer that needs that fact, split into prompt and reply."""
    name, food = pick.choice(NAMES), pick.choice(FOODS)
    return f"{name} eats {food}.<u>what does {name} eat?<a>", f"{food}{', thanks' if polite else ''}.{STOP}"


def chat_tensors(pairs, vocab):
    """Inputs and targets, padded on the right. −100 hides the prompt and the padding from the loss."""
    xs, ys = [], []
    for prompt, reply in pairs:
        ids = encode(prompt + reply, vocab)
        target = ids[1:].clone()
        target[: len(prompt) - 1] = -100  # only the reply is taught
        xs.append(F.pad(ids[:-1], (0, CONTEXT - len(ids) + 1), value=vocab[" "]))
        ys.append(F.pad(target, (0, CONTEXT - len(target)), value=-100))
    return torch.stack(xs), torch.stack(ys)


# @a:pretrain
def pretrain(vocab, steps=300):
    """The base model: predict the next character of plain text. It has never seen a question."""
    torch.manual_seed(0)
    model = TinyGPT(len(vocab), context=CONTEXT)
    x, y = windows(encode(make_corpus(1000), vocab), CONTEXT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(steps):
        batch = torch.randint(len(x), (32,))
        loss = F.cross_entropy(model(x[batch]).reshape(-1, len(vocab)), y[batch].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


# @a:fine-tune
def fine_tune(model, pairs, vocab, steps=80, lr=3e-3):
    """The same loss on the same model, now on conversations, with a gentler learning rate."""
    torch.manual_seed(1)
    x, y = chat_tensors(pairs, vocab)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        batch = torch.randint(len(x), (min(16, len(x)),))
        loss = F.cross_entropy(model(x[batch]).reshape(-1, len(vocab)), y[batch].reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


# @a:reply
@torch.no_grad()
def reply(model, prompt, vocab, limit=16):
    """Write until the stop token, or give up after `limit` characters."""
    ids, stop = encode(prompt, vocab), vocab[STOP]
    start = len(ids)
    for _ in range(limit):
        ids = torch.cat([ids, model(ids[-CONTEXT:])[-1:].argmax(-1)])
        if ids[-1] == stop:
            break
    return decode(ids[start:], vocab)


def correct_rate(model, vocab, n=50):
    """The share of unseen conversations answered exactly right, stop token included."""
    pick = random.Random(123)
    tests = [make_chat(pick) for _ in range(n)]
    return sum(reply(model, p, vocab) == r for p, r in tests) / n


# @a:demo
if __name__ == "__main__":
    # @a:vocab
    vocab = build_vocab(sorted(set(make_corpus(1000) + EXTRA)))
    pick = random.Random(0)
    few = [make_chat(pick) for _ in range(16)]  # chats are expensive to write, so there are few
    prompt, answer = few[0]
    print(f"a chat example: {prompt!r} → {answer!r}")

    # @a:before
    base = pretrain(vocab)
    base_rate = correct_rate(base, vocab)
    print(f"base model: {reply(base, prompt, vocab)!r}, correct {base_rate:.0%}")
    assert base_rate == 0  # it continues text; it doesn't answer

    # @a:after
    chat = fine_tune(base, few, vocab)
    rate = correct_rate(chat, vocab)
    print(f"fine-tuned: {reply(chat, prompt, vocab)!r}, correct {rate:.0%}")
    assert rate > 0.7

    # @a:scratch
    torch.manual_seed(0)
    scratch = fine_tune(TinyGPT(len(vocab), context=CONTEXT), few, vocab)
    scratch_rate = correct_rate(scratch, vocab)
    print(f"no pretraining, same 16 chats and steps: correct {scratch_rate:.0%}")
    assert rate > scratch_rate
