---
title: NumPy basics
summary: Arrays, shapes, vectorised operations and broadcasting, the toolkit every implementation on this site is written in.
area: foundations
---

NumPy stores numbers in **arrays**: blocks of same-typed values with a **shape**. A vector of
5 numbers has shape `(5,)`; a table of 100 rows and 3 features has shape `(100, 3)`. In
machine learning, `X` is almost always a 2-D array of shape `(n_rows, n_features)`.

Three ideas carry most of the code you'll read here:

- **Vectorisation.** `x * 2` or `x + y` works on whole arrays at once, element by element,
  in fast compiled code. Python loops over array elements are rarely needed.
- **Reductions along an axis.** `X.mean(axis=0)` averages *down* the rows (one value per
  column); `axis=1` works *across* each row.
- **Broadcasting.** When shapes differ, NumPy stretches size-1 dimensions to match:
  `X - X.mean(axis=0)` subtracts each column's mean from every row of that column, with no
  loop.

$$X \in \mathbb{R}^{n \times p}, \qquad \bar{x} \in \mathbb{R}^{p}, \qquad (X - \bar{x})_{ij} = X_{ij} - \bar{x}_j$$

Plus `@` for matrix products and boolean masks (`x[x > 0]`) to select rows. Get comfortable
with checking `.shape`: most bugs in numerical code are shape bugs.
