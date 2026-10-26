"""An LSTM in PyTorch: gates written out, checked against nn.LSTM, then trained beside an RNN that can't remember."""

import torch
import torch.nn.functional as F

# @a:reuse
from language_modeling_pytorch import FOODS, NAMES, generate, make_corpus
from recurrent_neural_networks_pytorch import CharRNN
from tokenization_pytorch import build_vocab, decode, encode


# @a:lstm
class CharLSTM(torch.nn.Module):
    """Token ids (…, T) → next-token logits (…, T, V), with a gated cell state beside the hidden state."""

    def __init__(self, V, d=64):
        super().__init__()
        self.embed = torch.nn.Embedding(V, d)
        self.x_gates = torch.nn.Linear(d, 4 * d)  # input, forget, candidate, output: all at once
        self.h_gates = torch.nn.Linear(d, 4 * d, bias=False)
        self.unembed = torch.nn.Linear(d, V)

    def forward(self, ids):
        x = self.x_gates(self.embed(ids))
        h = c = torch.zeros_like(x[..., 0, : x.shape[-1] // 4])
        states = []
        for t in range(x.shape[-2]):
            i, f, g, o = (x[..., t, :] + self.h_gates(h)).chunk(4, dim=-1)
            # @a:cell
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)  # keep some, write some
            h = torch.sigmoid(o) * torch.tanh(c)  # show some
            states.append(h)
        return self.unembed(torch.stack(states, dim=-2))


# @a:sentences
def sentences(text, ids, length):
    """Inputs and next-token targets for every window of `length` that starts a sentence."""
    starts = [0] + [i + 2 for i in range(len(text) - length - 3) if text[i : i + 2] == ". "]
    return (torch.stack([ids[s : s + length] for s in starts]),
            torch.stack([ids[s + 1 : s + length + 1] for s in starts]))


# @a:copies
@torch.no_grad()
def copies(model, vocab):
    """In how many of the 64 name/food sentences does the model repeat the right name after 'so '?"""
    prompts = [(name, encode(f"{name} likes {food}, so ", vocab)) for name in NAMES for food in FOODS]
    return sum(decode(generate(model, p, 3, temperature=1e-3), vocab).endswith(f"so {name}")
               for name, p in prompts)


# @a:demo
if __name__ == "__main__":
    # @a:check
    torch.manual_seed(0)
    text = make_corpus(1000)
    vocab = build_vocab(text)
    V = len(vocab)
    lstm = CharLSTM(V)
    reference = torch.nn.LSTM(64, 64, batch_first=True)  # PyTorch's own, same gate order: i, f, g, o
    with torch.no_grad():
        reference.weight_ih_l0.copy_(lstm.x_gates.weight)
        reference.bias_ih_l0.copy_(lstm.x_gates.bias)
        reference.weight_hh_l0.copy_(lstm.h_gates.weight)
        reference.bias_hh_l0.zero_()
        ids = encode(text[:30], vocab)[None]
        assert torch.allclose(lstm(ids), lstm.unembed(reference(lstm.embed(ids))[0]), atol=1e-6)
    print("the hand-written gates compute exactly what nn.LSTM does")

    # @a:train
    torch.set_num_threads(1)  # tiny tensors: threads cost more than they save
    ids = encode(text, vocab)
    split = int(0.9 * len(ids))
    x_train, y_train = sentences(text[:split], ids[:split], 34)
    copied = {}
    for name, Model in [("RNN", CharRNN), ("LSTM", CharLSTM)]:
        torch.manual_seed(0)  # the same start and the same batches for both
        model = Model(V)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for step in range(601):
            batch = torch.randint(len(x_train), (32,))
            loss = F.cross_entropy(model(x_train[batch]).reshape(-1, V), y_train[batch].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        copied[name] = copies(model, vocab)
        print(f"{name:>4}: loss {loss.item():.3f}, repeats the right name in {copied[name]} of 64")

    # @a:remembers
    assert copied["RNN"] <= 16  # same data, same steps: the RNN still guesses (8 is chance)...
    assert copied["LSTM"] >= 32  # ...while the gated cell carries the name across 17 characters
