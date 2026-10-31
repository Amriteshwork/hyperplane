---
title: Recurrent networks, from scratch
summary: From a NumPy array to an LSTM translator with attention. A plain RNN caught forgetting, a gated LSTM that remembers, then the wall that made attention necessary.
order: 2
goal: seq2seq-attention
chapters:
  - title: The maths you need
    summary: Arrays, vectors, matrix products, slopes and probabilities. Skim whatever you already know; everything later leans on these.
    concepts: [numpy-basics, vectors, matrix-multiplication, derivatives, probability, logarithms-exponentials, sigmoid, softmax]
  - title: How a model learns
    summary: A loss that says how wrong the model is, and gradient descent to make it less wrong, one small step at a time.
    concepts: [loss-function, mean-squared-error, normal-equation, gradient-descent, learning-rate, stochastic-gradient-descent, feature-expansion]
  - title: From classifiers to neural networks
    summary: Predict probabilities, draw boundaries, then stack neurons into layers and train them all with backpropagation.
    concepts: [log-loss, decision-boundary, neurons, activation-functions, backpropagation, adam-optimizer]
  - title: Text as numbers
    summary: Chop text into tokens, give each token a learned vector, and turn "predict the next token" into a loss.
    concepts: [tokenization, embeddings, language-modeling]
  - title: Memory
    summary: Carry a hidden state from token to token, train it by backpropagating through time, and see how far back it really remembers.
    concepts: [recurrent-neural-networks]
  - title: Gated memory
    summary: Add a memory cell and gates that decide what to keep, write and show, and the name finally survives.
    concepts: [lstm]
  - title: Attention
    summary: A token asks a question, every token offers an answer. Just enough attention to look back across two sentences.
    concepts: [self-attention, causal-masking, multi-head-attention, cross-attention]
  - title: The wall, and the way through
    summary: Two LSTMs translate through one vector until long inputs break them. Let the decoder look back, and the wall disappears.
    concepts: [seq2seq-attention]
---

Before transformers, language was modelled by **recurrent neural networks**. An RNN reads
text the way you do, one token at a time, and keeps a running summary of what it has read
so far. That one idea, a small network applied again and again with its output fed back in,
ran translation and speech recognition through the 2010s. Its weaknesses are the reason
attention was invented.

This track lays out **everything a recurrent network needs, in order**, from "what is a
NumPy array?" to an RNN with hand-written backpropagation through time. Then it puts the RNN
to a test: *"zoe likes jam, so "*, who eats? The RNN learns the sentence pattern quickly but
loses the name on the way, and the page shows why: vanishing gradients. The next chapter
adds the **LSTM**'s gated memory cell, trains it beside the RNN on the same batches, and it
remembers.

Then comes the problem an LSTM can't solve. Two LSTMs can translate, but only by squeezing the
whole sentence through one vector. The last chapters measure that wall and break through it with
**attention**, the idea that went on to replace recurrence altogether:

::figure[bottleneck-curve]

**How to use it**

- **Skip what you know.** The first four chapters are shared with the transformer track. If
  you've done that one, go straight to chapter 5.
- **Run the code.** Every page has a from-scratch implementation in NumPy and another in
  PyTorch, each checking its own result. "Download with dependencies" gives you a folder
  that runs on its own.
- **Then read the transformer track.** It takes the last chapter's idea all the way: if attention
  can look straight back at anything, the recurrence can go.
