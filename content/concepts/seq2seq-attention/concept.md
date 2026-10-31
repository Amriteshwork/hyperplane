---
title: Seq2seq with attention
summary: Two LSTMs translate by squeezing a whole sentence through one vector, and long sentences fall apart. Let the decoder look back at every encoder state and the problem vanishes. Attention began here.
area: sequence-models
prerequisites: [lstm, cross-attention]
videos:
  one-vector-problem:
    title: Why one vector isn't enough
    youtube: mNrgIjAo9Mg
---

The LSTM page ended with a claim: encoder-decoder LSTMs squeezed a whole sentence into one
vector, and that was a bottleneck. This page **measures** that claim, then fixes it.

### Translating with two LSTMs

In 2014, Sutskever et al. and Cho et al. built translators from two recurrent networks. The
**encoder** reads the English, one word at a time. The **decoder** starts from the encoder's
final hidden state and writes the German, one word at a time.

Everything the decoder knows about the English is in that one vector. Twelve words or
fifty, it's the same size. Imagine reading a long sentence once, handing over a single
sticky note, and asking a colleague to translate from the note alone.

::video[one-vector-problem]

### See the wall

The test is a translation that needs no cleverness: shopping lists, word by word. *"figs jam
rice"* becomes *"feigen marmelade reis"*. The word order never changes and every word has
one translation, so the only challenge is **remembering a list**. Train both models with the
same LSTM size, data and number of steps. The only difference is attention:

::figure[bottleneck-curve]

These are real results from this page's PyTorch code: lists it never saw, with every word
required to be right. Without attention it's perfect on short lists and falls to nothing by
12 words. It didn't run out of skill; it ran out of room.

### The fix: look back

Don't throw the encoder's states away. Keep all of them, $h_1, \dots, h_S$, one for each
English word. Before writing each German word, the decoder scores every one, using its
own current state $s$ as the question:

$$e_i = v^\top \tanh(W_s\, s + W_h\, h_i), \qquad \alpha_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}, \qquad c = \sum_i \alpha_i\, h_i.$$

- $e_i$ is a **score**: how useful is English word $i$ right now?
- $\alpha_i$ turns the scores into **weights** that add up to 1.
- $c$ is the **context**, a blend of encoder states made fresh for this word.

The context goes into the decoder beside the previous word. Nobody says which word to read.
Here's what the trained model actually does, weights included:

::figure[attention-lookback]

The diagonal appears because the list keeps its order, and gradient descent found it: looking
at the right English word lowers the loss. On a real language pair, it bends and crosses
where word order differs.

### You've seen this before

This is [cross-attention](/concepts/cross-attention) with different parts. The question comes
from the decoder, the answers come from the encoder, and a softmax blends them.

| | Bahdanau (2014) | Cross-attention (2017) |
|---|---|---|
| query | decoder's LSTM state $s$ | decoder's token vector, via $W_Q$ |
| keys and values | encoder's LSTM states | encoder's vectors, via $W_K, W_V$ |
| score | additive: $v^\top\tanh(W_s s + W_h h)$ | scaled dot product: $QK^\top/\sqrt{d_k}$ |
| heads | one | many |

Luong et al. (2015) soon showed a simpler dot-product score works well, and that's the
version the transformer scaled up. *Attention Is All You Need* then asked the obvious next
question: if attention is doing the remembering, what's the LSTM still for? Its answer was to
remove the recurrence, use attention for everything, and let all the words run at once.

**Two differences from the paper.** Bahdanau et al. used a **bidirectional** encoder, one LSTM
reading each way, so each $h_i$ knows about the words on both sides. Ours reads one way,
which is enough for a list. They also trained on real English and French, where the same
effect appears at a much larger scale.

**What the code shows.** The NumPy file runs an LSTM encoder and `additive_attention`, and
checks that the plain model's input is 8 numbers for 4 words or 40, and that the context is a
weighted sum of all the states. The PyTorch file trains both translators in about 40 seconds
and prints the table behind the plot and the weights behind the figure.
