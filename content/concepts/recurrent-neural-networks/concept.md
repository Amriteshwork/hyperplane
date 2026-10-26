---
title: Recurrent neural networks
summary: Read text one token at a time, carrying a hidden state that sums up everything so far. The same small network at every step, trained by backpropagation through time.
area: sequence-models
prerequisites: [embeddings, language-modeling, backpropagation, adam-optimizer]
---

The bigram on the language-modeling page sees one character and nothing else. To predict
well, a model needs **memory**: some summary of everything it has read. A **recurrent
neural network** (RNN) keeps one. It's a vector $h_t$, the **hidden state**, updated once per
token:

$$h_t = \tanh\big(E[x_t] + h_{t-1} W + b\big), \qquad \text{logits}_t = h_t U + c.$$

$E[x_t]$ is the current token's embedding, $h_{t-1}$ is the memory so far, and the new state
mixes the two through one layer of neurons. Start from $h_0 = 0$ and repeat to the end of
the text. Each $\text{logits}_t$ is a prediction for token $t+1$, scored with the same
cross-entropy as any language model.

**One network, used again and again.** $E$, $W$, $b$, $U$ and $c$ are the *same* at every
step. Only $h$ changes. So the model can read text of any length with a fixed number of
weights, and what it learns about "after *so*" applies wherever "so" appears.

**Unroll it, and it's an ordinary deep network.** Draw one copy of the layer per token, with
$h$ flowing from each copy to the next. Twenty tokens make a twenty-layer network whose
layers all share their weights. That makes training ordinary too: **backpropagation through
time** (BPTT) is backpropagation on the unrolled network. Run forward over the whole text,
keeping every $h_t$. Then walk from the last token back to the first, applying the chain
rule once per step.

**BPTT, worked out.** Name the pre-activation $z_t = E[x_t] + h_{t-1}W + b$, so
$h_t = \tanh(z_t)$, and let the loss be the average cross-entropy over all $T$ predictions.
Vectors are rows, as in the code. Four facts give every gradient.

1. **At each output.** Softmax plus cross-entropy has the simplest possible gradient, the
   same one as logistic regression:
   $$g_t = \frac{\partial L}{\partial\, \text{logits}_t} = \frac{1}{T}\big(p_t - \text{onehot}(y_t)\big).$$
   In the code that's `d_logits[t]`.
2. **At each state, two sources.** $h_t$ feeds this step's prediction *and* the next state,
   so its gradient is the sum of both paths:
   $$\frac{\partial L}{\partial h_t} = g_t\, U^\top + \delta_{t+1} W^\top,
   \qquad \delta_{t+1} = \frac{\partial L}{\partial z_{t+1}}.$$
   The second term is `d_h_next`: everything the future has to say about $h_t$. At the last
   step there is no future, so it starts at 0.
3. **Back through tanh.** $\tanh'(z) = 1 - \tanh^2(z)$, and $h_t$ is already stored:
   $$\delta_t = \frac{\partial L}{\partial h_t} \odot \big(1 - h_t^2\big).$$
   That's `dz`. Multiplying it by $W^\top$ gives the next `d_h_next`, one step further back.
4. **Shared weights add up.** $W$ is used at every step. Picture $T$ copies $W_1, \dots, W_T$
   that happen to be equal: the chain rule gives each copy its own gradient, and the real
   $W$ gets their sum. Each copy's gradient is its input times the error at its output, as in
   any layer:
   $$\frac{\partial L}{\partial W} = \sum_t h_{t-1}^\top\, \delta_t, \qquad
   \frac{\partial L}{\partial b} = \sum_t \delta_t, \qquad
   \frac{\partial L}{\partial U} = \sum_t h_t^\top g_t, \qquad
   \frac{\partial L}{\partial c} = \sum_t g_t,$$
   and the embedding row of each token read collects its step's $\delta_t$:
   `dE[ids[t]] += dz`.

Facts 2 and 3 together say how a gradient travels back one step when nothing new is
added: it is multiplied by

$$\frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}\big(1 - h_t^2\big)\, W^\top.$$

The whole backward pass costs about as much as the forward pass, but it needs every $h_t$
kept in memory, which grows with the length of the text. The NumPy code checks its
hand-written BPTT against a finite difference. Step through it on a one-unit RNN:

::figure[bptt-unrolled]

**The catch: vanishing and exploding gradients.** A signal from $k$ steps back reaches the
loss through $k$ of those Jacobians multiplied together. A product of many matrices behaves
like a power. If $W$ shrinks vectors, the product shrinks geometrically and the far past
stops influencing learning. That's a **vanishing gradient**. If $W$ stretches them, the
product blows up: an **exploding gradient**. In the code, halving $W$ scales a gradient 24
steps back by about $10^{-6}$; tripling it scales it by about $2000$. Two standard
patches:

- **Gradient clipping** caps a gradient's size before each step, so one explosion can't wreck
  the weights. It does nothing for vanishing.
- **Truncated BPTT** backpropagates through only the last few dozen steps, to keep training
  cheap. Dependencies longer than that can't be learnt directly.

**Watching it forget.** Both files train on the language-modeling page's text, *"zoe likes
jam, so zoe eats jam."* The RNN learns everything local, the spelling and the sentence
pattern, and gets to about a quarter of the bigram's loss. Then ask it for the name after
"so ": it's right in 8 of 64 sentences, exactly what blind guessing among eight names gives.
The name is 17 characters back, and by then it has faded from $h$. The transformer, trained
on the same text, the same windows and the same number of steps, gets 62 of 64.

**What came next.** The **LSTM** (1997) and the simpler **GRU** (2014) add *gates*: learned
switches that decide what to keep in memory and what to overwrite. They carry an extra state
along a path that is mostly *added to* rather than multiplied, the same idea as a residual
connection, so gradients survive far longer. The LSTM page trains one beside this RNN on
the same text, and it remembers the name. Gated RNNs ran machine translation and speech
recognition through the 2010s. Two problems remained:

1. **No parallelism.** $h_t$ needs $h_{t-1}$, so a 1,000-token text takes 1,000 steps one
   after another, however many processors you have.
2. **A bottleneck.** Everything the model knows about the past must squeeze into one
   fixed-size vector.

**Attention** removes both. Every token looks directly at every earlier token, all at once,
with no single vector in between. That's why transformers replaced RNNs for language. RNN
ideas are still around, though: recent "linear RNN" and state-space models (Mamba, RWKV)
bring back a recurrent state for fast generation over very long texts.
