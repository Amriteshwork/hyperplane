---
title: Word embeddings (word2vec)
summary: Hide the meaning, keep the neighbours. Train vectors so that words predict the words around them, and meaning appears as geometry, without anyone labelling anything.
area: sequence-models
prerequisites: [embeddings, sigmoid, log-loss, stochastic-gradient-descent]
videos:
  sliding-window:
    title: One sentence becomes a dataset
    youtube: wzkPKIhKOs4
---

The [Embeddings](/concepts/embeddings) page said a network can learn vectors where similar
words sit together. Its code used vectors made by hand. This page shows how to get them
from nothing but raw text, using one of the most surprising ideas in machine learning.

### You shall know a word by the company it keeps

What's a *wampimuk*? No idea. But read *"the furry wampimuk sleeps in a tree"* and *"we saw
a wampimuk hunting at night"*, and you've got a good guess: a small animal. You learnt that
from its neighbours. So give a machine only one job:

> **Given a word, predict the words near it.**

That task is fake: nobody needs its answers. We keep the **vectors** it learns along the way
and throw the task away. This is **skip-gram**, from word2vec (Mikolov et al., 2013).

::video[sliding-window]

### One trick makes it fast

A true skip-gram would softmax over the whole vocabulary for every pair, which is far too
slow. **Negative sampling** swaps it for a much easier question. For a real pair, say "yes".
For $k$ random pairs, say "no":

$$\mathcal L = -\log \sigma(u_{\text{word}} \cdot v_{\text{neighbour}}) \;-\; \sum_{n=1}^{k} \log \sigma(-u_{\text{word}} \cdot v_{n}).$$

Read it as two forces:

- **The pull.** A word's vector swings toward the vectors of words it really appears with.
- **The push.** It swings away from random words.

Why do "cat" and "dog" end up together, when they may never appear side by side? Because they
get pulled toward **the same neighbours**, like "sleeps" and "runs". Two words with similar
company land in similar places. Watch it happen live, with two-dimensional vectors so you
can see them:

::figure[skipgram-trainer]

The colours are only there for you. The model never sees them.

### Now the real thing

Here are real vectors, from **GloVe**: a close cousin of word2vec, trained on billions of words
of text. They have 50 dimensions, drawn here in two. The neighbour lists and the arithmetic
below use all 50:

::figure[word-map]

**King − man + woman ≈ queen.** The step from *man* to *woman* points roughly the same way as
*king* to *queen*. No one programmed gender, or capitals, or comparatives. They appeared
because those relationships shape which words appear together.

Three cautions, so you don't over-believe it:

- The search usually **excludes the three input words**. Without that, the nearest word is
  often just "king".
- It's not perfect. In these same vectors, *walk → walked* as *go → ?* puts "got" above
  "went". Word arithmetic works best on the famous examples.
- Vectors learn **the text's biases** too, so they've been shown to pick up gender stereotypes
  about jobs. A model reflects its data.

### One vector per word, and why that isn't the end

Word2vec gives "bank" one vector, whether it's beside a river or holding your money. That's
the gap that [self-attention](/concepts/self-attention) fills, by letting each token's vector
absorb its context.

The core idea didn't go away, though. A transformer's embedding table is trained the same way
in spirit: pushed around by a next-token loss until tokens that act alike sit together. The
difference is that it's trained along with the rest of the model.

**What the code shows.** The NumPy file writes the loss and its gradients by hand, checks them
against finite differences, and trains on 1,500 sentences in a few seconds. Every word's nearest
neighbours end up in its own group, though nothing in the data names the groups. The PyTorch
file batches the same thing with `nn.Embedding` and flattens the result to 2-D with PCA.
