---
title: Translation, from scratch
summary: From a NumPy array to a model that translates. Build a GPT first, then give it a second sequence to read, and pair it with an encoder, the way the original transformer was built.
order: 4
goal: encoder-decoder
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
  - title: Reading another sentence
    summary: Cross-attention. The same attention, but the questions come from one sentence and the answers from another, so each word being written can look up the words it translates.
    concepts: [cross-attention]
  - title: Encoder and decoder
    summary: An encoder reads the English once, a decoder writes the German word by word and reads the encoder at every step. Train it, and watch an alignment appear that nobody taught it.
    concepts: [encoder-decoder]
---

A GPT **writes**: it continues whatever text it's given. A translator has a harder job. It
has to write one sentence while **reading another**, and the words don't come in the same
order. *"so raj eats figs"* is *"also isst raj feigen"* in German: the verb jumps ahead of
its subject. The transformer was invented for exactly this job, in 2017, and its original
form has two halves: an **encoder** that reads and a **decoder** that writes.

This track builds it. The first seven chapters are **the transformer track**, from "what
is a NumPy array?" to a GPT you train yourself. Then two short chapters turn it into a
translator:

- **Reading another sentence.** Cross-attention: the attention you already know, with the
  keys and values taken from a different sequence. Each German word asks the English
  sentence which word it should come from.
- **Encoder and decoder.** Run the GPT's blocks with no mask over the English, give each
  decoder block a cross-attention step into them, and train on sentence pairs. The tiny
  model you train translates sentences it has never seen, verb order included.

::figure[cross-attention]

**How to use it**

- **Done the transformer track?** Go straight to chapter 8. Everything before it is the
  same.
- **Run the code.** The encoder-decoder's NumPy file imports the GPT's own `block` for the
  encoder and adds a single `cross_attention` call to make the decoder. That's the
  clearest way to see how little is new.
- **Then read the history.** [Seq2seq with attention](/concepts/seq2seq-attention) shows
  where cross-attention came from: a fix for translators that squeezed a whole sentence into
  one vector. The *Attention Is All You Need* (2017) walkthrough shows the encoder-decoder
  transformer itself.
