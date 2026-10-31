---
title: From GPT to chat model
summary: A tiny GPT only continues text. Learn how to pick its words, train it safely, teach it to answer, and shape it with human votes. Everything between a transformer and a chatbot.
order: 5
goal: rlhf
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
  - title: Picking the next word
    summary: A model only gives you odds. Temperature, top-k, top-p and beam search decide what you actually read, and a cache makes it fast.
    concepts: [text-generation]
  - title: Training at scale
    summary: Warmup, clipping, dropout and shared weights, each tested on a real run, including when they didn't help.
    concepts: [llm-training]
  - title: Teaching it to answer
    summary: A conversation is one long sequence. Fine-tune on example chats, with a loss that only counts the assistant's words.
    concepts: [chat-fine-tuning]
  - title: Shaping it with votes
    summary: Train a reward model on "A beats B", push the chat model towards high scores, and find out why you mustn't push too hard.
    concepts: [rlhf]
---

The transformer page ends with a tiny GPT and a promise: *from here to ChatGPT is mostly scale
and data, then a second phase.* This track is that road. Ask a model that's just finished
pretraining a question, and it doesn't answer. It continues the text, perhaps with more questions.
Four steps change that.

- **Picking the next word.** A model gives odds, not words. Choosing from them is a separate
  decision, and you can measure what each choice costs.
- **Training at scale.** A few cheap safety tricks keep long runs from falling apart. We test each
  one honestly, and some don't help on a small model.
- **Teaching it to answer.** With no new architecture, a base model that scores 0% on questions
  learns to reply and to stop, from just 16 example conversations.
- **Shaping it with votes.** People compare replies, a reward model learns from those votes, and
  the chat model shifts towards them. Push too hard and real quality collapses while the score
  keeps rising:

::figure[rlhf-leash]

**How to use it**

- **Done the transformer track?** Go straight to chapter 8. Everything before it is the same.
- **Run the code.** Each page trains something real on a laptop CPU, and each number on these pages
  comes from code you can run.
