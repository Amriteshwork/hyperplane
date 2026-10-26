---
title: Recurrent networks, from scratch
summary: From a NumPy array to an LSTM. Every idea a recurrent network needs, then a plain RNN caught forgetting and the gated LSTM that remembers.
order: 2
goal: lstm
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
---

Before transformers, language was modelled by **recurrent neural networks**. An RNN reads
text the way you do, one token at a time, and keeps a running summary of what it has read
so far. That one idea, a small network applied again and again with its output fed back in,
ran translation and speech recognition through the 2010s. Its weaknesses are the reason
attention was invented.

This track lays out **everything a recurrent network needs, in order**, from "what is a
NumPy array?" to an RNN with hand-written backpropagation through time. Then it puts the RNN
to a test: *"zoe likes jam, so "*, who eats? The RNN learns the sentence pattern quickly but
loses the name on the way, and the page shows why: vanishing gradients. The last chapter
adds the **LSTM**'s gated memory cell, trains it beside the RNN on the same batches, and it
remembers.

**How to use it**

- **Skip what you know.** The first four chapters are shared with the transformer track. If
  you've done that one, go straight to chapter 5.
- **Run the code.** Every page has a from-scratch implementation in NumPy and another in
  PyTorch, each checking its own result. "Download with dependencies" gives you a folder
  that runs on its own.
- **Then read the transformer track.** It solves the same test a different way: instead of
  carrying the name along, attention looks straight back at it.
