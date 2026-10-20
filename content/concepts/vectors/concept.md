---
title: Vectors and the dot product
summary: A vector is a list of numbers with a direction and a length; the dot product measures how much two vectors agree.
area: foundations
prerequisites: [numpy-basics]
---

A **vector** is an ordered list of numbers, $v = (v_1, \dots, v_n)$. Geometrically it's an
arrow from the origin; in machine learning it's usually one example's features or a model's
weights.

The **dot product** multiplies matching entries and adds them up:

$$u \cdot v = \sum_{i=1}^{n} u_i v_i = \lVert u \rVert \, \lVert v \rVert \cos\theta$$

- Positive: the vectors point roughly the same way. Zero: they're **perpendicular**.
  Negative: roughly opposite.
- A linear model's prediction is a dot product: $\hat y = w \cdot x + b$.

The **norm** (length) of a vector is $\lVert v \rVert = \sqrt{v \cdot v}$, Pythagoras in
$n$ dimensions. Norms measure the size of errors ($\lVert y - \hat y \rVert$) and of weights
(the L2 penalty is $\lVert w \rVert^2$). Dividing the dot product by both lengths gives the
**cosine similarity**, used to compare embeddings.
