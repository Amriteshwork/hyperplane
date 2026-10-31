---
title: Lazy learning
summary: Skip the training. Keep the examples, and do the work when a question arrives by looking at the most similar ones.
area: instance-based
prerequisites: [distance-metrics, train-test-split, overfitting-underfitting]
---

Most models are **eager**: training boils the data down to a few parameters, then throws it
away. A linear model with ten features keeps eleven numbers, however many rows it saw.

A **lazy** learner does the opposite. Training stores the rows. All the work happens when a
query arrives, by searching the stored rows for the most similar ones. The best-known example
is **k-nearest neighbours**:

1. Measure the distance from the query to every training row.
2. Take the $k$ closest.
3. Classification: return their most common label. Regression: return their average.

$$\hat y(x) = \frac{1}{k}\sum_{i \in N_k(x)} y_i$$

Try it: move the query, change $k$, and see who votes.

::figure[knn-playground]

## k is the bias–variance dial

$k$ is the one choice that really matters, because it sets how local the answer is.

- $k = 1$ memorises. Every training row is its own nearest neighbour, so training accuracy is
  100% whatever the data, noise included. In this page's code it scores 100% on training rows
  and 71% on new ones: **high variance**.
- A moderate $k$ averages the noise away. At $k = 15$ the same data scores 81% on both.
- $k = n$ gives every query the same answer, the majority class or the overall mean, which
  scores 50% here: **high bias**.

So the training score can't choose $k$, and you need a validation set or cross-validation.

## What laziness costs

| | Eager (linear model) | Lazy (k-NN) |
|---|---|---|
| Training | does the work | stores the rows |
| Prediction | one dot product | a search of the rows |
| Model size | a few numbers | the whole training set |
| New data | usually retrain | just add the rows |
| Shape of the fit | fixed by the formula | whatever the data shows |

A brute-force search costs $O(nd)$ for each query, with $n$ rows and $d$ features. Search
structures (**KD trees** and **ball trees**) avoid measuring most of those distances, but they
lose their advantage as $d$ grows.

Saving the model saves the data. If the training rows are private, the model file is private
too, which isn't true of a linear model's few coefficients.

It also has no parameters to read, but a prediction is easy to explain: "these five rows were
the most similar, and this is what happened to them."
