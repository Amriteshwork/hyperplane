"""An encoder-decoder transformer in PyTorch: train it to translate English into German, then watch where it looks."""

import random
import re

import torch
import torch.nn.functional as F

# @a:reuse
from causal_masking_pytorch import causal_mask
from cross_attention_pytorch import cross_attention
from language_modeling_pytorch import FOODS, NAMES
from layer_normalization_pytorch import layer_norm
from multi_head_attention_pytorch import multi_head_attention
from positional_encoding_pytorch import sinusoidal
from tokenization_pytorch import build_vocab, encode
from transformer_pytorch import Block

# @a:pairs
GERMAN = {"likes": "mag", "so": "also", "eats": "isst", "figs": "feigen", "plums": "pflaumen",
          "rice": "reis", "nuts": "nüsse", "soup": "suppe", "kale": "grünkohl", "eggs": "eier",
          "jam": "marmelade"}


def words(text):
    """Word-level tokens: 'zoe likes jam, so' → ['zoe', 'likes', 'jam', ',', 'so']."""
    return re.findall(r"\w+|[,.]", text)


def to_german(english):
    """'zoe likes jam, so raj eats figs.' → 'zoe mag marmelade, also isst raj feigen.' as tokens."""
    out = [GERMAN.get(w, w) for w in words(english)]
    i = out.index("also")
    out[i + 1], out[i + 2] = out[i + 2], out[i + 1]  # after "also", German puts the verb first
    return out


def make_pairs(n, seed=0):
    """`n` English sentences with their German translations, as word lists."""
    pick = random.Random(seed)
    pairs = []
    for _ in range(n):
        a, b, f, g = pick.choice(NAMES), pick.choice(NAMES), pick.choice(FOODS), pick.choice(FOODS)
        english = f"{a} likes {f}, so {b} eats {g}."
        pairs.append((words(english), to_german(english)))
    return pairs


# @a:decoder-block
class DecoderBlock(Block):
    """The transformer's block plus cross-attention: look back at the German, read the English, think."""

    def __init__(self, d, n_heads):
        super().__init__(d, n_heads, causal=True)
        self.cross = torch.nn.Parameter(torch.randn(4, d, d) / d**0.5)  # its own W_Q, W_K, W_V, W_O
        self.ln_x = torch.nn.Parameter(torch.stack([torch.ones(d), torch.zeros(d)]))

    def forward(self, x, memory):
        x = x + multi_head_attention(layer_norm(x, *self.ln1), *self.attn, self.n_heads,
                                     causal_mask(x.shape[-2]))[0]
        read, self.weights = cross_attention(layer_norm(x, *self.ln_x), memory, *self.cross, self.n_heads)
        x = x + read
        return x + self.mlp(layer_norm(x, *self.ln2))


# @a:model
class TinySeq2Seq(torch.nn.Module):
    """English ids (…, S) and German ids so far (…, T) → next German word scores (…, T, V_tgt)."""

    def __init__(self, V_src, V_tgt, d=32, n_heads=4, n_layers=2, context=16):
        super().__init__()
        self.src_embed, self.tgt_embed = torch.nn.Embedding(V_src, d), torch.nn.Embedding(V_tgt, d)
        self.register_buffer("positions", sinusoidal(context, d))
        self.encoder = torch.nn.Sequential(*[Block(d, n_heads, causal=False) for _ in range(n_layers)])
        self.decoder = torch.nn.ModuleList([DecoderBlock(d, n_heads) for _ in range(n_layers)])
        self.ln_enc, self.ln_dec = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.unembed = torch.nn.Linear(d, V_tgt)

    def encode(self, src):
        return self.ln_enc(self.encoder(self.src_embed(src) + self.positions[: src.shape[-1]]))

    def decode(self, tgt, memory):
        y = self.tgt_embed(tgt) + self.positions[: tgt.shape[-1]]
        for b in self.decoder:
            y = b(y, memory)
        return self.unembed(self.ln_dec(y))

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src))


# @a:translate
@torch.no_grad()
def translate(model, src, bos, eos, max_len=16):
    """Greedy decoding for a batch of sentences: encode once, then add the likeliest word, n times."""
    memory = model.encode(src)
    out = torch.full((len(src), 1), bos)
    for _ in range(max_len - 1):
        out = torch.cat([out, model.decode(out, memory)[:, -1:].argmax(-1)], dim=1)
        if (out == eos).any(dim=1).all():
            break
    return out[:, 1:]


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    pairs = make_pairs(3000)
    train, test = pairs[:2500], [p for p in make_pairs(200, seed=1) if p not in pairs][:100]
    src_vocab = build_vocab(sorted({w for en, _ in pairs for w in en}))
    tgt_vocab = build_vocab(sorted({w for _, de in pairs for w in de}))
    tgt_vocab |= {"[BOS]": len(tgt_vocab), "[EOS]": len(tgt_vocab) + 1}
    BOS, EOS = tgt_vocab["[BOS]"], tgt_vocab["[EOS]"]

    def tensors(batch):
        """English ids, German input ([BOS] + words) and German target (words + [EOS])."""
        src = torch.stack([encode(en, src_vocab) for en, _ in batch])
        de = torch.stack([encode(de, tgt_vocab) for _, de in batch])
        return src, F.pad(de, (1, 0), value=BOS), F.pad(de, (0, 1), value=EOS)

    # @a:train
    model = TinySeq2Seq(len(src_vocab), len(tgt_vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(151):
        src, tgt_in, tgt_out = tensors(random.Random(step).sample(train, 64))
        loss = F.cross_entropy(model(src, tgt_in).flatten(0, 1), tgt_out.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:3d}: loss {loss.item():.3f}")

    # @a:unseen
    src, tgt_in, tgt_out = tensors(test)
    out = translate(model, src, BOS, EOS, tgt_out.shape[1] + 1)
    exact = (out == tgt_out).all(dim=1).float().mean().item()
    itos = {i: w for w, i in tgt_vocab.items()}
    for (en, _), ids in zip(test[:3], out):
        print(f"  {' '.join(en):38} → {' '.join(itos[i] for i in ids.tolist() if i != EOS)}")
    print(f"{exact:.0%} of {len(test)} sentences it never saw translated exactly")
    assert exact > 0.9

    # @a:alignment
    with torch.no_grad():
        model(src, tgt_in)
    reads = model.decoder[-1].weights.mean(dim=1).argmax(dim=-1)  # (sentences, T): most-read English word
    slots = {0: 0, 2: 2, 6: 5, 7: 7}  # German position of each name and food → its English position
    en, de = test[0]
    print("  " + "  ".join(f"{de[g]}←{en[reads[0, g]]}" for g in slots))
    hits = sum(reads[:, g].eq(e).float().mean().item() for g, e in slots.items()) / len(slots)
    print(f"names and foods read their own English word {hits:.0%} of the time")
    assert hits > 0.75  # an alignment nobody taught it
