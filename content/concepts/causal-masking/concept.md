---
title: Causal masking
summary: Forbid every token from looking at the tokens after it, so a model can learn to predict the future without peeking at it.
area: sequence-models
prerequisites: [self-attention]
---

A language model's job is to predict the next token. Say we train it on "the cat sat". At
the position of "cat" it should predict "sat". But plain self-attention lets "cat" look at
**every** token, including "sat" itself. The model would learn to copy the answer from one
position ahead, reach zero training loss, and be useless the moment it has to generate
text, where the future doesn't exist yet. It's like an exam where the answer key is
stapled to the back.

**The fix is one triangle.** Before the softmax, set every score where a token would look
*ahead* to $-\infty$:

$$\text{score}_{ts} = \begin{cases} q_t\cdot k_s/\sqrt{d_k} & s \le t \\ -\infty & s > t\end{cases}
\qquad\Longrightarrow\qquad a_{ts} = 0 \text{ for every } s > t.$$

Since $e^{-\infty} = 0$, softmax hands out all the weight among the past and the present.
The mask is the lower triangle of a $T\times T$ matrix:

$$M = \begin{pmatrix} 1&0&0&0\\ 1&1&0&0\\ 1&1&1&0\\ 1&1&1&1 \end{pmatrix}
\qquad (\text{row} = \text{who's looking},\ \text{column} = \text{at whom}).$$

Row 1 can only see itself, so its output is just its own value. Row 4 sees everything
before it. Information now flows in one direction only: **left to right, like time**.
The name comes from that: an output can only be *caused* by earlier inputs.

Turn the mask on and off and watch the top-right triangle disappear:

::figure[attention-heatmap]

**The hidden superpower: T lessons per sentence.** Because no position can see its
answer, we can train all positions **at once**. One pass over a 1,000-token window gives
1,000 next-token predictions, each a fair test: predict token 2 from token 1, token 3 from
tokens 1–2, and so on. That's the shifted-by-one target from tokenization, and it's what
makes training GPT-style models so efficient.

**A beautiful special case.** If every query is zero, all allowed scores are equal and
each position simply **averages everything so far**: a running mean. Real attention
learns *weighted* averages instead: "for predicting this token, the noun three words back
matters most". Karpathy's *Let's build GPT* starts from exactly this running-mean trick,
and the code below checks it.

**Where you'll meet it.** GPT, Llama, Claude and every other text *generator* are
"decoder-only" transformers: causal masking in every layer. BERT-style *encoders* (used for
classifying or searching text) skip the mask, because they're allowed to read the whole
sentence before answering.
