---
title: Multi-head attention
summary: Run several small attentions side by side, each free to look for something different, then combine what they found.
area: sequence-models
prerequisites: [causal-masking]
---

One attention pattern is one way of looking. But reading even a short sentence needs many
at once. In *"The cat that the dogs chased **was** tired"*, to handle "was" you want to
know the subject (cat, singular, so "was" not "were"), the previous word (chased), and
whether you're inside a clause. A single softmax row has to *average* all of that into one
blend, and averaging blurs.

**So use several heads.** Instead of one attention over $d$-dimensional queries and keys,
run $h$ attentions over $d/h$ dimensions each, every one with its own projections. Each
**head** produces its own attention pattern, and its own answer. Then glue the answers side
by side and mix them with one more matrix $W_O$:

$$\text{head}_i = \text{Attention}(XW_Q^{(i)},\, XW_K^{(i)},\, XW_V^{(i)}), \qquad
\text{MHA}(X) = [\text{head}_1 \,|\, \cdots \,|\, \text{head}_h]\; W_O .$$

**It's free.** Stack the per-head matrices side by side and $W_Q^{(1)}, \dots, W_Q^{(h)}$ is just
one $d\times d$ matrix whose columns are cut into $h$ blocks. So multi-head attention has
**exactly the same parameters as single-head attention** ($4d^2$, counting $W_O$) and
about the same compute. The only change is that the softmax runs separately on each block.
The PyTorch code computes all heads in one batched matrix product by reshaping $(T, d)$
into $(h, T, d/h)$.

**Heads specialise.** Nobody assigns the jobs; they emerge in training because different
patterns help predict different things. In trained models, researchers find heads that
reliably:

- look at the **previous token**,
- connect a **verb to its subject** or a pronoun to what it refers to,
- copy from earlier in the text: *"…Mr Dursley… Mr D"* → the head that finds the last "D"
  and looks at what followed it (an **induction head**, believed to drive much of
  in-context learning).

The code builds the first of these **by hand**. Each token is just its position, and the
two heads' keys are wired so that head 0 finds the previous position and head 1 finds
itself. Same input, same formula, two completely different behaviours, decided entirely
by $W_K$.

**Typical sizes:** the original transformer used $d = 512$ with $h = 8$ heads of 64
dimensions. GPT-3 used $d = 12{,}288$ with 96 heads of 128. Heads stay smallish, and models
grow by adding more of them.

Multi-head attention is the **communication** step of a transformer: tokens exchange
information. Each block then gives every token a private **computation** step (an MLP), wraps both in
**residual connections** and **layer normalization**, and stacks the blocks deep. That's the
transformer.
