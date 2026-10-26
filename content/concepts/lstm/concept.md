---
title: LSTM
summary: A recurrent network with a separate memory cell and three gates that decide what to forget, what to write and what to show. Gradients ride the cell back in time, so it can remember far longer than a plain RNN.
area: sequence-models
prerequisites: [recurrent-neural-networks]
---

The RNN page ended with a failure. On *"zoe likes jam, so zoe eats jam."* a plain RNN learnt
the spelling and the sentence pattern, then forgot the name 17 characters later. The
reason: every step multiplies the memory by $W$ and squashes it through tanh, so a gradient
from far back shrinks geometrically before it can teach anything. The **long short-term
memory** (LSTM, Hochreiter & Schmidhuber, 1997) fixes this by changing *how memory is
stored*.

**Two states instead of one.** Beside the hidden state $h_t$, an LSTM carries a **cell
state** $c_t$, its long-term memory. Each step computes four vectors from the current input
and the previous hidden state, each with its own weights:

$$\begin{aligned}
i_t &= \sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i) && \textit{input gate: how much to write}\\
f_t &= \sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f) && \textit{forget gate: how much to keep}\\
g_t &= \tanh(x_t W_{xg} + h_{t-1} W_{hg} + b_g) && \textit{candidate: what to write}\\
o_t &= \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o) && \textit{output gate: how much to show}
\end{aligned}$$

and then

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t).$$

The gates are sigmoids, so every entry is between 0 (shut) and 1 (open), and $\odot$ multiplies
entry by entry. Each gate is a learned, per-unit, per-token *decision*. Reading "zoe", the
input gate can open and write the name into a few cells. Over the next 17 characters the
forget gates of those cells can stay near 1 and the input gates near 0: hold. At "so ", the
output gate opens and the name reaches $h$, where the read-out can use it.

**Why the gradient survives.** Follow the cell back one step: $\partial c_t / \partial c_{t-1}$
along the cell's own path is just $\operatorname{diag}(f_t)$. No $W$, no tanh slope. Over $k$
steps that's $\prod f$, so when the network learns to hold, $f \approx 1$, the gradient arrives
almost intact. Hochreiter called this the **constant error carousel**. It's the same idea as
a residual connection, $x + F(x)$: memory is *added to* rather than rewritten, so there is a
path that carries gradients without shrinking. The NumPy code measures it. With forget
gates near ½ (the default when biases start at 0), 24 steps keep $10^{-7}$ of a gradient.
With the forget bias at 5 they keep 84%, and a change to the first character is still
visible in the cell 24 steps later.

**Watching it remember.** The PyTorch file trains a plain RNN and an LSTM on the same
text, with the same windows (each starting at a sentence), the same batches and the same
600 Adam steps. The RNN still picks the right name after "so " in only 8 of 64 sentences,
which is chance. The LSTM picks it in most of them: its gates learnt to hold the name.

**Practical notes**

- **Forget-gate bias.** A common trick (Gers et al., 2000; Jozefowicz et al., 2015) is to start
  $b_f$ at 1, so the network remembers by default and learns what to forget. PyTorch's
  `nn.LSTM` starts all biases near 0.
- **The GRU** (Cho et al., 2014) is a lighter variant with two gates. An *update* gate $z$
  blends old and new, $h_t = (1 - z)\odot h_{t-1} + z \odot \tilde h_t$, playing both forget and
  input gate. It has no separate cell and usually performs about as well, with fewer
  weights.
- **Cost.** Four gates mean four times the weights and work of a plain RNN with the same
  $d$. And it's still a loop: $c_t$ needs $c_{t-1}$, so a long text is read one step at a
  time.

**Where LSTMs went.** Through the 2010s, stacked LSTMs ran machine translation, speech
recognition and handwriting recognition. Encoder–decoder LSTMs squeezed a whole sentence
into one vector, which was a bottleneck. **Attention** was first added to fix it, letting the
decoder look back at every encoder state. Transformers then kept the attention and dropped
the recurrence altogether, which removed the step-by-step loop. LSTMs remain a good choice
for small streaming problems, like sensor data or audio on a device, where one step at a
time with a fixed-size memory is exactly what you want.
