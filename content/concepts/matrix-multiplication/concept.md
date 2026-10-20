---
title: Matrix multiplication
summary: Every entry of a matrix product is a dot product of a row with a column; it's how a model applies its weights to a whole dataset at once.
area: foundations
prerequisites: [vectors]
---

A **matrix** is a grid of numbers with shape $m \times n$. The product of an $m \times n$
matrix $A$ and an $n \times p$ matrix $B$ is the $m \times p$ matrix whose entries are dot
products of $A$'s rows with $B$'s columns:

$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj} = (\text{row } i \text{ of } A) \cdot (\text{column } j \text{ of } B)$$

The inner dimensions must match ($n$), and the outer ones give the result's shape.

Why it's everywhere in ML:

- $X w$ applies a weight vector to every row of a dataset at once: all predictions of a
  linear model in one product.
- $X^\top X$ (the **transpose** $X^\top$ swaps rows and columns) appears in the normal
  equation; $X W$ with a weight *matrix* is a neural-network layer.

Rules worth knowing: $AB \ne BA$ in general; $(AB)^\top = B^\top A^\top$; the **identity**
$I$ (ones on the diagonal) leaves any matrix unchanged, $AI = A$. A matrix also *transforms*
space, stretching and rotating vectors, and multiplying two matrices chains the
transformations.
