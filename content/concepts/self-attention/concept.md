---
title: Self-attention
summary: Every token asks a question, every token offers an answer, and each token's new vector is a blend of the answers that best match its question.
area: sequence-models
prerequisites: [embeddings, softmax]
---

Read this sentence: *"The animal didn't cross the street because **it** was too tired."*
What does "it" mean? You knew instantly: the animal. Now change one word: *"…because
**it** was too wide."* Now "it" is the street. The embedding of "it" is the same row of
the table both times. **To understand a token, you have to look at the other tokens.**
Self-attention is a learnable way of looking.

**A soft dictionary lookup.** Think of a Python dictionary. You bring a **query**, compare
it with every **key**, and get back the **value** of the key that matches. Attention does
the same thing with vectors, and softly. Each token produces three vectors from its
embedding $x_t$, using three learned matrices:

- a **query** $q_t = x_t W_Q$: *"what am I looking for?"* (for "it": "a noun that could get tired");
- a **key** $k_t = x_t W_K$: *"what do I contain?"* (for "animal": "I'm a noun, I'm alive");
- a **value** $v_t = x_t W_V$: *"what I'll pass on if you pick me."*

Token $t$ scores every token $s$ by how well its query matches their key (a dot product),
turns the scores into weights with softmax, and takes the weighted average of the values:

$$\text{score}_{ts} = \frac{q_t \cdot k_s}{\sqrt{d_k}}, \qquad
a_{ts} = \operatorname{softmax}_s(\text{score}_{ts}), \qquad
\text{out}_t = \sum_s a_{ts}\, v_s .$$

Do it for all tokens at once by stacking rows into matrices, and you get the most
important equation in modern AI:

$$\boxed{\;\text{Attention}(Q, K, V) = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)\,V\;}$$

$QK^\top$ is a $T \times T$ table of every query against every key. Softmax turns each
row into weights that sum to 1 (the **attention pattern**). Multiplying by $V$ blends the
values. Each output row is a new, **context-aware** version of that token: "it" comes out
carrying a bit of "animal".

Here it is on a sentence whose queries and keys were picked by hand, so you can read them:

::figure[attention-heatmap]

**Why divide by $\sqrt{d_k}$?** If the entries of $q$ and $k$ are random with variance 1,
their dot product is a sum of $d_k$ such terms, so its variance is $d_k$. With $d_k = 512$,
scores are typically around ±22, and softmax of numbers that far apart is essentially
one-hot. Then the gradients through softmax vanish and learning stalls. Dividing by
$\sqrt{d_k}$ brings the variance back to 1. The code measures this directly.

**Two surprising properties**, both shown in the code:

- **Every token talks to every token in one step.** An RNN passes information along the
  sentence one word at a time; attention connects word 1 to word 500 directly. It is also
  all matrix products, which GPUs devour. That's why transformers replaced RNNs. The cost:
  the $T \times T$ table grows with the square of the length.
- **Attention has no idea of word order.** Shuffle the tokens and the outputs are shuffled
  the same way, otherwise unchanged. "Dog bites man" and "man bites dog" look identical!
  **Positional encoding** fixes this later.

The parameters are just $W_Q, W_K, W_V$. Attention itself has nothing to learn; it's a fixed
recipe for mixing. Everything a transformer knows about *which* tokens should look at
*which* is stored in those three matrices.
