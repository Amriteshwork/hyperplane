---
title: Transformer
summary: Stack blocks of "tokens talk" (attention) and "tokens think" (MLP) on a residual stream, end with a prediction for the next token, and train on text. That's GPT.
area: sequence-models
prerequisites: [multi-head-attention, positional-encoding, layer-normalization, language-modeling, adam-optimizer]
---

Every piece is now on the table. This page snaps them together into a working **GPT**, a
decoder-only transformer, the architecture behind ChatGPT, Claude, Llama and Gemini. The
surprise, once you've seen the parts, is how *little* glue it needs.

**The whole model, top to bottom:**

1. **Tokenize** the text into ids.
2. **Embed** each id ($E$, a $V\times d$ table) and **add its position**: $X_0 = E[\text{ids}] + P$.
3. **Repeat $L$ times**, a transformer **block**:
   $$X \leftarrow X + \text{MultiHeadAttention}\big(\text{LN}(X)\big) \qquad \textit{tokens talk}$$
   $$X \leftarrow X + \text{MLP}\big(\text{LN}(X)\big) \qquad\qquad\qquad\ \textit{each token thinks}$$
4. **Normalise** once more and **unembed**: $\text{logits} = \text{LN}(X)\,W_{\text{out}}$, one
   score per vocabulary token, **at every position**.
5. Train with **cross-entropy** against the text shifted by one. Generate by sampling the
   last position's prediction and appending it.

Click through it, at GPT-2's and GPT-3's real sizes:

::figure[transformer-block]

**Talk, then think.** The attention sub-layer is the only place tokens exchange
information, and the causal mask makes that information flow strictly forward. The MLP
(two linear layers with a non-linearity, widened 4× in the middle) works on every token
**separately**. It's the same tiny network from the Neurons page, applied at each position.
Researchers think much of a model's factual knowledge lives in these MLPs. Each block
therefore alternates *gather context* and *process it*, and stacking blocks lets the model
build up from letters to words to meaning.

**Why it works when it's deep:** residual connections give the gradient an express lane,
and pre-LN gives every sub-layer a standardised input. Without both, a stack of dozens of
blocks wouldn't train.

**Where the parameters are.** Per block: $4d^2$ in attention ($W_Q, W_K, W_V, W_O$) and
$8d^2$ in the MLP ($d \to 4d \to d$), so about $12d^2$. GPT-2 small has $d = 768$ and
$L = 12$: $12 \times 12 \times 768^2 \approx 85$M in its blocks, plus a
$50{,}257 \times 768$ embedding table (39M) for 124M total. GPT-3 is the same recipe with
$d = 12{,}288$ and $L = 96$: 175 billion. **Scaling is mostly turning up $d$ and $L$.**

**See it learn something a bigram can't.** The PyTorch code trains a two-block model
(about 27,000 parameters, about 20 seconds on a laptop CPU) on the generated text from the
Language-modeling page: *"zoe likes jam, so zoe eats jam."* Its validation loss falls
from $\log V \approx 3.2$ to about 0.27, against the bigram's 1.32. Prompted with *"raj
likes figs, so "*, it writes *"raj eats figs."*. It has learnt to **look back and copy**,
the job of the induction heads described on the Multi-head attention page. A bigram would
pick a random name. What's left of the loss is mostly the names and foods themselves,
which are random and can't be predicted.

**What the NumPy version shows.** A forward pass built *only* from earlier concepts'
functions (`embed`, `sinusoidal`, `multi_head_attention`, `causal_mask`, `layer_norm`,
`residual`), checked for three things that must hold in any GPT:

- changing late tokens never changes earlier predictions (causal, end to end);
- an untrained model's loss is $\log V$, uniform guessing, as it should be;
- the parameter count matches the $12d^2$-per-block formula exactly.

**From here to ChatGPT** is mostly scale and data, then a second phase. Pre-training a much
bigger version of this model on a large slice of the internet gives a model that continues
text. **Fine-tuning** on conversations and learning from human (or AI) feedback turns
"continue this text" into "answer this helpfully". The architecture barely changes. There
are refinements (RoPE, RMSNorm, other activation functions, more efficient attention), but
the block on this page is still recognisably the block inside today's models.
