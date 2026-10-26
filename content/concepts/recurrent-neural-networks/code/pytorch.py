"""A recurrent network in PyTorch: the loop written out, checked against nn.RNN, then trained, and caught forgetting."""

import torch
import torch.nn.functional as F

# @a:reuse
from language_modeling_pytorch import FOODS, NAMES, fit_bigram, generate, make_corpus
from tokenization_pytorch import build_vocab, decode, encode, windows


# @a:rnn
class CharRNN(torch.nn.Module):
    """Token ids (…, T) → next-token logits (…, T, V), reading left to right with one hidden state."""

    def __init__(self, V, d=64):
        super().__init__()
        self.embed = torch.nn.Embedding(V, d)  # plays the role of W_xh
        self.W = torch.nn.Linear(d, d)  # h_{t−1} → h_t, shared by every step
        self.unembed = torch.nn.Linear(d, V)

    def forward(self, ids):
        x = self.embed(ids)
        h, states = torch.zeros_like(x[..., 0, :]), []
        for t in range(x.shape[-2]):
            h = torch.tanh(x[..., t, :] + self.W(h))
            states.append(h)
        return self.unembed(torch.stack(states, dim=-2))


# @a:demo
if __name__ == "__main__":
    # @a:check
    torch.manual_seed(0)
    text = make_corpus(1000)
    vocab = build_vocab(text)
    V = len(vocab)
    model = CharRNN(V)
    reference = torch.nn.RNN(64, 64, batch_first=True)  # PyTorch's own: tanh(x W_ihᵀ + b_ih + h W_hhᵀ + b_hh)
    with torch.no_grad():
        reference.weight_ih_l0.copy_(torch.eye(64))
        reference.bias_ih_l0.zero_()
        reference.weight_hh_l0.copy_(model.W.weight)
        reference.bias_hh_l0.copy_(model.W.bias)
        ids = encode(text[:30], vocab)[None]
        states, _ = reference(model.embed(ids))
        assert torch.allclose(model(ids), model.unembed(states), atol=1e-6)
    print("the hand-written loop computes exactly what nn.RNN does")

    # @a:train
    ids = encode(text, vocab)
    split = int(0.9 * len(ids))
    x_train, y_train = windows(ids[:split], 24)
    x_val, y_val = windows(ids[split:], 24)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(601):
        batch = torch.randint(len(x_train), (32,))
        loss = F.cross_entropy(model(x_train[batch]).reshape(-1, V), y_train[batch].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tame exploding gradients
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:3d}: loss {loss.item():.3f}")

    # @a:evaluate
    with torch.no_grad():
        every = slice(None, None, 24)
        val_loss = F.cross_entropy(model(x_val[every]).reshape(-1, V), y_val[every].reshape(-1)).item()
    bigram_loss = -fit_bigram(ids[:split], V)[ids[split:-1], ids[split + 1 :]].mean().item()
    print(f"validation loss: RNN {val_loss:.3f} vs bigram {bigram_loss:.3f}")
    assert val_loss < bigram_loss / 3

    # @a:forgets
    copied = 0
    with torch.no_grad():
        for name in NAMES:
            for food in FOODS:
                prompt = encode(f"{name} likes {food}, so ", vocab)
                copied += decode(generate(model, prompt, 3, temperature=1e-3), vocab).endswith(f"so {name}")
    print(f"it repeats the right name in {copied} of 64 sentences; guessing gets 8")
    assert copied <= 16  # 17 characters back is too far: the name has faded from h
