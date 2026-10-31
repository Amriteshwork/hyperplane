"""Word2vec in PyTorch: the same skip-gram with negative sampling, batched, then squeezed to 2-D with PCA to look at."""

import random

import torch
import torch.nn.functional as F

# @a:reuse
from tokenization_pytorch import build_vocab

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
    pick = random.Random(seed)
    sentences = []
    for _ in range(n):
        group = pick.choice(list(GROUPS))
        sentences.append(pick.choice(PATTERNS[group]).format(pick.choice(GROUPS[group])).split())
    return sentences


# @a:pairs
def skipgram_pairs(sentences, vocab, window=2):
    """(word, a word within `window` of it), for every position of every sentence."""
    pairs = [(vocab[words[i]], vocab[words[j]])
             for words in sentences for i in range(len(words))
             for j in range(max(0, i - window), min(len(words), i + window + 1)) if j != i]
    return torch.tensor(pairs)


# @a:model
class SkipGram(torch.nn.Module):
    """Two tables: each word as a centre (`center`) and as a neighbour (`context`)."""

    def __init__(self, V, d=16):
        super().__init__()
        self.center, self.context = torch.nn.Embedding(V, d), torch.nn.Embedding(V, d)
        for table in (self.center, self.context):
            torch.nn.init.normal_(table.weight, std=0.1)

    # @a:loss
    def forward(self, centers, contexts, negatives):
        """Real pairs should score high, random ones low: binary cross-entropy on dot products."""
        u = self.center(centers)
        pos = (u * self.context(contexts)).sum(-1)
        neg = (self.context(negatives) @ u[..., None])[..., 0]
        return F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + \
            F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))


# @a:pca
def pca_2d(E):
    """The two directions along which the vectors vary most; project every word onto them."""
    centered = E - E.mean(dim=0)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ Vh[:2].T


# @a:demo
if __name__ == "__main__":
    # @a:train
    torch.manual_seed(0)
    sentences = make_sentences(1500)
    vocab = build_vocab(sorted({w for s in sentences for w in s}))
    pairs = skipgram_pairs(sentences, vocab)
    model = SkipGram(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(1501):
        batch = pairs[torch.randint(len(pairs), (256,))]
        loss = model(batch[:, 0], batch[:, 1], torch.randint(len(vocab), (256, 4)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step:4d}: loss {loss.item():.3f}")

    # @a:result
    E = model.center.weight.detach()
    unit = F.normalize(E, dim=-1)
    S = unit @ unit.T  # cosine similarity of every pair of words
    names = list(vocab)
    for word in ("cat", "soup", "tram"):
        ranked = S[vocab[word]].argsort(descending=True)[1:4]
        print(f"  nearest to {word}: {[names[i] for i in ranked]}")
    groups = list(GROUPS.values())
    for words in groups:
        others = [w for g in groups if g is not words for w in g]
        same = torch.stack([S[vocab[a], vocab[b]] for a in words for b in words if a != b]).mean().item()
        different = torch.stack([S[vocab[a], vocab[b]] for a in words for b in others]).mean().item()
        print(f"  same group {same:.2f}, other groups {different:.2f}")
        assert same > different + 0.3  # nobody told it the groups

    # @a:map
    xy = pca_2d(E)
    for group, words in GROUPS.items():
        print(f"  {group:8}" + "  ".join(f"{w} ({xy[vocab[w], 0]:+.1f}, {xy[vocab[w], 1]:+.1f})" for w in words))
