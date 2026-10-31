"""Seq2seq with attention in PyTorch: the same LSTM translator with and without it, measured on longer and longer inputs."""

import random

import torch
import torch.nn.functional as F

# @a:reuse
from language_modeling_pytorch import FOODS

# @a:data
GERMAN = dict(zip(FOODS, ["feigen", "pflaumen", "reis", "nüsse", "suppe", "grünkohl", "eier", "marmelade"]))
VOCAB = ["[BOS]", "[EOS]", *FOODS, *GERMAN.values()]
ID = {w: i for i, w in enumerate(VOCAB)}
BOS, EOS = ID["[BOS]"], ID["[EOS]"]


def batch(length, size, pick):
    """`size` shopping lists of one `length`, so no padding: ids of English, German in and German out."""
    english = [[pick.choice(FOODS) for _ in range(length)] for _ in range(size)]
    src = torch.tensor([[ID[w] for w in row] for row in english])
    de = torch.tensor([[ID[GERMAN[w]] for w in row] for row in english])
    return src, F.pad(de, (1, 0), value=BOS), F.pad(de, (0, 1), value=EOS)


# @a:model
class Translator(torch.nn.Module):
    """An LSTM reads the English and an LSTM writes the German, with or without attention between them."""

    def __init__(self, attend, d=64):
        super().__init__()
        self.attend, wide = attend, 2 * d if attend else d  # attention feeds a context vector in too
        self.embed = torch.nn.Embedding(len(VOCAB), d)
        self.encoder = torch.nn.LSTM(d, d, batch_first=True)
        self.decoder = torch.nn.LSTMCell(wide, d)
        self.W_s, self.W_h = torch.nn.Linear(d, d, bias=False), torch.nn.Linear(d, d, bias=False)
        self.v = torch.nn.Linear(d, 1, bias=False)
        self.out = torch.nn.Linear(wide, len(VOCAB))

    # @a:additive
    def context(self, s, H):
        """Bahdanau's additive attention: score every encoder state against the decoder's state, then blend."""
        scores = self.v(torch.tanh(self.W_s(s)[:, None] + self.W_h(H)))[..., 0]  # (B, S)
        weights = torch.softmax(scores, dim=-1)
        return (weights[:, None] @ H)[:, 0], weights

    # @a:forward
    def forward(self, src, tgt=None, steps=None):
        """With `tgt`, teacher forcing. Without it, feed back the model's own best guess for `steps` steps."""
        H, (s, c) = self.encoder(self.embed(src))
        s, c = s[0], c[0]  # the final state: all the plain model ever knows about the English
        word, logits, alignment = torch.full((len(src),), BOS), [], []
        for t in range(tgt.shape[1] if tgt is not None else steps):
            x = self.embed(tgt[:, t] if tgt is not None else word)
            if self.attend:
                ctx, weights = self.context(s, H)
                alignment.append(weights)
                x = torch.cat([x, ctx], dim=-1)
            s, c = self.decoder(x, (s, c))
            logits.append(self.out(torch.cat([s, ctx], dim=-1) if self.attend else s))
            word = logits[-1].argmax(-1)
        return torch.stack(logits, 1), alignment


# @a:fit
def fit(attend, steps=600):
    """Same seed, same data, same steps: the only difference is attention."""
    torch.manual_seed(0)
    model = Translator(attend)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        pick = random.Random(step)
        src, tgt_in, tgt_out = batch(pick.randint(1, 12), 32, pick)
        loss = F.cross_entropy(model(src, tgt_in)[0].flatten(0, 1), tgt_out.flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # recurrent gradients can explode
        optimizer.step()
    return model


# @a:score
@torch.no_grad()
def exact_rate(model, length, size=200):
    """The share of unseen lists translated with every single word right."""
    src, tgt_in, tgt_out = batch(length, size, random.Random(1000 + length))
    guess = model(src, steps=tgt_in.shape[1])[0].argmax(-1)
    return (guess == tgt_out).all(dim=1).float().mean().item()


# @a:demo
if __name__ == "__main__":
    # @a:compare
    plain, attentive = fit(False), fit(True)
    lengths = (1, 2, 3, 4, 5, 6, 8, 10, 12)
    print("words:         " + " ".join(f"{n:4d}" for n in lengths))
    rates = {}
    for name, model in (("no attention", plain), ("attention", attentive)):
        rates[name] = [exact_rate(model, n) for n in lengths]
        print(f"{name:14} " + " ".join(f"{r:4.0%}" for r in rates[name]))
    assert rates["no attention"][0] > 0.9  # short lists fit in one vector
    assert rates["no attention"][-1] < 0.1  # long ones don't
    assert rates["attention"][-1] > rates["no attention"][-1] + 0.3

    # @a:alignment
    english = ["figs", "jam", "rice", "kale", "eggs", "soup"]
    src = torch.tensor([[ID[w] for w in english]])
    with torch.no_grad():
        logits, alignment = attentive(src, steps=len(english))
    weights = torch.cat(alignment)  # (German words, English words)
    print("German word reads:")
    for t, row in enumerate(weights):
        print(f"  {VOCAB[logits[0, t].argmax()]:10} " + " ".join(f"{w:.2f}" for w in row.tolist()))
    diagonal = (weights.argmax(-1) == torch.arange(len(english))).float().mean().item()
    print(f"each German word looks hardest at its own English word {diagonal:.0%} of the time")
    assert diagonal >= 0.8  # an alignment nobody taught it
