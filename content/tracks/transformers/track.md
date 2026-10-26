---
title: Transformers, from scratch
summary: From a NumPy array to a working GPT, one concept at a time. Every idea a transformer needs, in an order where nothing is used before it's explained.
order: 1
goal: transformer
chapters:
  - title: The maths you need
    summary: Arrays, vectors, matrix products, slopes and probabilities. Skim whatever you already know; everything later leans on these.
    concepts: [numpy-basics, vectors, matrix-multiplication, derivatives, probability, logarithms-exponentials, sigmoid, softmax]
  - title: How a model learns
    summary: A loss that says how wrong the model is, and gradient descent to make it less wrong, one small step at a time.
    concepts: [loss-function, mean-squared-error, normal-equation, gradient-descent, learning-rate, stochastic-gradient-descent, train-test-split, feature-scaling, feature-expansion]
  - title: From classifiers to neural networks
    summary: Predict probabilities, draw boundaries, then stack neurons into layers and train them all with backpropagation.
    concepts: [log-loss, decision-boundary, neurons, activation-functions, backpropagation, adam-optimizer]
  - title: Text as numbers
    summary: Chop text into tokens, give each token a learned vector, and turn "predict the next token" into a loss.
    concepts: [tokenization, embeddings, language-modeling]
  - title: Attention
    summary: The big idea. Every token asks a question, every token offers an answer, and context flows between them.
    concepts: [self-attention, causal-masking, multi-head-attention, positional-encoding]
  - title: Going deep
    summary: The two tricks that let dozens of blocks be stacked and still train.
    concepts: [residual-connections, layer-normalization]
  - title: The transformer
    summary: Snap every piece together into a tiny GPT, train it, and watch it learn something no simpler model can.
    concepts: [transformer]
---

ChatGPT, Claude, Llama, Gemini: under the hood, every one of them is a **transformer**. It's
the most important architecture in modern AI, and it's often presented as a diagram of
mysterious boxes. It doesn't have to be. A transformer is built from a few ideas, each small
enough to understand completely, and most of them are ideas this site already teaches for
simpler models.

This track lays out **all of them, in order**. It starts at "what is a NumPy array?" and
ends with a GPT you've trained yourself, which learns to complete *"raj likes figs, so "*
with *"raj eats figs"*. It's a small model, but it does something no simpler model on the
way can. Every page lists only things that came before it.

**How to use it**

- **Skip what you know.** Each chapter says what it's for. If you already know gradient
  descent, jump ahead. The learning path at the top of every page shows what that page
  builds on, so you can always step back.
- **Run the code.** Every page has a from-scratch implementation in NumPy and another in
  PyTorch, each checking its own result. Later files import
  earlier ones: the transformer at the end is assembled from the attention, layer-norm and
  embedding code you'll have read on the way. "Download with dependencies" gives you a
  folder that runs on its own.
- **Play with the figures.** The interactive pictures are there to build intuition before
  the maths. Here's one from the Attention chapter: click a word and see what it looks at.

::figure[attention-heatmap]

**Where's the hard part?** Chapters 1–3 are the foundations of all of machine learning, not
just transformers. If they're new, they're most of the work, and worth it. Chapter 5,
attention, is the genuinely new idea. Give it the most time. After that, the last two
chapters are short: putting it together is easier than you'd expect.
