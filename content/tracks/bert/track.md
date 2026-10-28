---
title: BERT, from scratch
summary: From a NumPy array to a model that reads both ways. Build a GPT first, then take off its mask, teach it to fill in blanks, and fine-tune it for any task.
order: 3
goal: bert
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
  - title: Fill in the blanks
    summary: A new pretraining task. Hide some words and predict them from the text on both sides, so the model can read in both directions without cheating.
    concepts: [masked-language-modeling]
  - title: BERT
    summary: The same blocks with the causal mask off, [CLS] and [SEP] around the input, and a small head per task. Pretrain once, fine-tune many times.
    concepts: [bert]
---

GPT **writes**. BERT **reads**. They're built from the same blocks, but GPT is only
allowed to look left, so it can generate text one token at a time. BERT looks both ways,
so it understands a sentence the way you do, all of it at once. That makes it the model
behind search ranking, text classification and the retrieval half of many chatbots.

This track builds both. The first seven chapters are **the transformer track**, from "what
is a NumPy array?" to a GPT you train yourself. Then two short chapters turn it into BERT:

- **Fill in the blanks.** Next-token prediction can't work once a model sees the future,
  so BERT learns by filling gaps instead: *"[MASK] likes figs, so raj eats figs."*
- **BERT.** Take the GPT's block, pass it a mask that hides nothing but padding, add
  `[CLS]`, `[SEP]` and segment embeddings, and pretrain. The tiny BERT you train answers
  **raj**, reading the clue from the right, where a GPT could only guess.

::figure[fill-in-the-blank]

**How to use it**

- **Done the transformer track?** Go straight to chapter 8. Everything before it is the
  same.
- **Run the code.** BERT's NumPy file imports the GPT's own `block` and hands it a
  different mask. That's the clearest way to see that the two models share their blocks
  and differ only in their mask.
- **Then read the paper.** The BERT walkthrough in Papers goes through the original
  section by section, with animations of masking, the input and fine-tuning.
