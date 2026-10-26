---
title: Language modeling
summary: Predict the next token, over and over. Score the predictions with cross-entropy, and generate text by sampling from them one token at a time.
area: sequence-models
prerequisites: [tokenization, log-loss, softmax, probability]
---

Every chatbot you've ever used is, at its core, an answer to one question: **given the
text so far, what comes next?** A **language model** outputs a probability for every token
in the vocabulary:

$$P(x_{t+1} \mid x_1, \dots, x_t).$$

That's all. Repeat it and you can score any text, by chaining the predictions:

$$P(x_1, \dots, x_T) = \prod_{t} P(x_{t+1} \mid x_{\le t}),$$

or **write** new text, by sampling a token, appending it, and asking again.

**The simplest language model: a bigram.** Ignore everything except the current token. Count
how often each character follows each other character in a training text, and divide by the
row totals. Done: a $V\times V$ table of probabilities. Adding 1 to every count first
(**Laplace smoothing**) keeps unseen pairs from getting probability 0, which would make the
loss infinite.

**Scoring: cross-entropy.** Look up the probability the model gave to the token that
*actually* came next, and take $-\log$:

$$\mathcal L = -\frac1N \sum_{t} \log P(x_{t+1} \mid x_{\le t}).$$

That's the log-loss from logistic regression with $V$ classes instead of two. With two
classes it's literally the same number, which the code checks. Some reference points:

- A model that guesses uniformly scores $\log V$. With our 24 characters, $\log 24 = 3.18$.
- A perfect model scores 0.
- $e^{\mathcal L}$ is the **perplexity**: "the model is as unsure as if it were choosing
  uniformly among this many tokens". Our bigram gets about 3.8: much better than 24.

**Sampling and temperature.** To generate, turn the scores into probabilities with softmax
and draw. Dividing the scores by a **temperature** $\tau$ first changes the mood:

$$p_k \propto \exp(z_k / \tau).$$

$\tau \to 0$ always picks the most likely token: safe, repetitive, often stuck in a loop
("eats eats eats"). $\tau = 1$ samples the model's honest beliefs. $\tau > 1$ flattens
everything toward uniform: creative, then nonsense. Your chatbot's "creativity" slider is
this one number.

**The bigram's blind spot.** The training text for this chain is generated:
*"zoe likes jam, so zoe eats jam."*, with names and foods chosen at random. After "so ",
which name comes next? Obviously the one from the start of the sentence. The bigram can't
know: all it sees is the space. So it spreads its bet over all eight names and is usually
wrong. No amount of counting fixes this. Predicting well needs **looking back** at the
right earlier token, wherever it is. That's exactly what attention does. The **transformer**
page trains a model on this same text and watches it learn to copy the name.

**Why this simple objective is so powerful.** To predict the next word of a murder mystery's
last page, you'd need to have worked out the culprit. Prediction pressure pushes a model to
learn spelling, grammar, facts and some reasoning, because each of them makes the next
token less surprising. Every text on the internet is free training data, since the labels
are just the text shifted by one.
