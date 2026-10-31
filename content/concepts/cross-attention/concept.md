---
title: Cross-attention
summary: Attention between two sequences. One asks the questions, the other supplies the answers, so a translation can look up the words of the sentence it's translating.
area: sequence-models
prerequisites: [multi-head-attention]
---

Self-attention lets the words of **one** sentence read each other. But a translator works
with **two** sentences: the English one on the page and the German one being written. To
write each German word, you look back at the English and find the word or words it should
come from. When you write "noir" in *le chat noir*, you're reading "black", even though
"black" came **before** "cat" in English.

**Cross-attention is that look-up.** It's the attention you already know, with one
change: the **queries** come from one sequence, and the **keys and values** come from
another.

$$\text{CrossAttention}(Y, X) = \text{softmax}\!\left(\frac{(YW_Q)(XW_K)^\top}{\sqrt{d_k}}\right) XW_V$$

- $Y$ ($T \times d$) is the sequence **asking**: the German written so far.
- $X$ ($S \times d$) is the sequence **answering**: the English sentence, as the
  encoder's vectors.
- Each German word's query is scored against every English word's key. The softmax turns
  those scores into weights, and the word receives a blend of the English **values**.

Set $Y = X$ and this is exactly self-attention. That's the whole difference: **where the
keys and values come from**.

Click a German word to see which English words it reads:

::figure[cross-attention]

### The shapes tell the story

Self-attention's weight matrix is square, $T \times T$, every token against every token.
Cross-attention's is **rectangular**, $T \times S$: one row per target token, one column
per source token.

- The output has the **target's** shape, $T \times d$. However long the source sentence
  is, each target token gets back one vector of size $d$.
- The two sequences can be **any lengths**. A 5-word English sentence can become a 7-word
  German one.
- Every row sums to 1: each target token spreads 100% of its attention over the source.

**No causal mask is needed here.** The source sentence is given in full before
translation starts, so every target token may read all of it, including the last word.
The decoder's *own* self-attention still has the causal mask, since it mustn't peek at
German words it hasn't written yet. The only mask cross-attention uses hides **padding**,
when sentences of different lengths share a batch.

### The weights are an alignment

Translators call the map from target words to source words an **alignment**. Old
statistical translation systems learnt it as a separate step. Cross-attention learns it
for free: the weights *are* a soft alignment, and nobody tells the model which words go
together. It discovers that because it has to translate.

The alignment is mostly **diagonal**, since word order is usually similar, with crossings
wherever the languages disagree: adjectives after nouns in French, the verb jumping ahead
of its subject in German after "also". It's **soft**: to choose between French *le*, *la*
and *l'*, the model can read both "the" and the noun after it.

### Where it came from

Cross-attention is older than self-attention. In 2014, Bahdanau, Cho and Bengio added it
to recurrent translation models, which until then squeezed the whole English sentence
into **one** vector before writing a word. Long sentences fell apart. Letting the decoder
look back at **every** source word fixed that, and they called the mechanism
*attention*. Three years later, *Attention Is All You Need* kept this layer (it calls it
**encoder-decoder attention**) and replaced the recurrence with self-attention.
[Seq2seq with attention](/concepts/seq2seq-attention) builds the 2014 version and measures
what it fixed.

It isn't only for translation. Image captioning reads image patches. Whisper turns speech
into text by reading audio frames. Stable Diffusion's image generator reads the words of
your prompt. In each case, cross-attention is how **one stream of data reads another**.

**What the code shows.** The NumPy `cross_attention` is the multi-head attention code with
`X @ W_K` and `X @ W_V` in place of `Y @ W_K` and `Y @ W_V`. The demo checks that
cross-attention of a sequence with itself gives exactly `multi_head_attention`. It builds
the *le chat noir* alignment by hand, with "noir" reading "black" out of order. It also
shows that the target tokens' reads are independent. The PyTorch version matches
`nn.MultiheadAttention` called with different query and key inputs, and shows that a
padded source with its padding masked gives exactly the same answer.
