---
title: KNN Classifier
summary: Predict a class by majority vote among the most similar training rows. No formula and no training, just a search and a show of hands.
difficulty: 1
---

## Intuition

To label a new row, find the $k$ most similar rows you've already seen and let them vote. That
is the whole algorithm. There are no weights and no equation of a boundary: the model is the
training data.

Logistic Regression draws one straight line between the classes. k-NN draws no line at all, yet
a boundary appears wherever the vote changes, and it can take any shape.

Move the query and change $k$. At $k = 1$ the boundary wraps around every stray point; at larger
$k$ it smooths out.

::figure[knn-playground]

## How it works

1. Preprocess and **standardise** the features. This step is essential here.
2. "Train" by storing the rows and their labels. Nothing else is computed, apart from a search
   tree to speed up later lookups.
3. For each query, apply the same preprocessing and measure the distance to the stored rows.
4. Take the $k$ nearest and count their votes, optionally weighting closer rows more.
5. Predict the class with the most votes. Each class's share of the votes is its "probability".

## Math

With $N_k(x)$ the $k$ training rows closest to $x$, the probability of class $c$ is its share of
the votes:

$$\hat P(c \mid x) = \frac{1}{k}\sum_{i \in N_k(x)} \mathbb 1[y_i = c] \qquad \text{or, with weights,} \qquad \hat P(c \mid x) = \frac{\sum_{i \in N_k(x)} \mathbb 1[y_i = c] / d(x, x_i)}{\sum_{i \in N_k(x)} 1 / d(x, x_i)}.$$

The prediction is $\arg\max_c \hat P(c \mid x)$. A tied vote goes to the first class in sorted order.

This estimates the true class probabilities directly from a small neighbourhood. As the data
grows (with $k$ growing too, but more slowly than $n$), k-NN approaches the best possible
classifier. Even 1-NN's error ends up at most twice the best achievable (Cover & Hart, 1967).

A brute-force search costs $O(nd)$ per query, with $n$ rows and $d$ features.

## Assumptions

- **Smoothness:** nearby rows tend to share a class.
- **Meaningful distance:** every feature, once scaled, matters about equally. An irrelevant
  feature adds noise to every distance.
- **Enough rows for the dimension:** the rows must fill the space densely enough to find
  neighbours that are actually close (see the curse of dimensionality).
- **Reasonably balanced classes:** a rare class gets outvoted in most neighbourhoods.

## Hyperparameters

| Setting | Effect |
|---|---|
| `n_neighbors` ($k$) | Small: jagged boundary, low bias, high variance. Large: smooth boundary, high bias, low variance. Start at 5, then use the cross-validated plot in the results. Odd values avoid ties between two classes. |
| `weights` | `uniform` gives each neighbour one vote. `distance` gives closer rows more say, which reduces the harm of a large $k$ and helps a rare class. |
| `metric` | `euclidean` (the default), `manhattan` (less swayed by one large difference) or `cosine` (direction, not length). |

Measured on the sample datasets (test accuracy, standardised features):

| Dataset | $k=1$ | $k=5$ | $k=15$ | Logistic Regression |
|---|---|---|---|---|
| Breast Cancer | 0.94 | 0.95 | 0.96 | 0.97 |
| Wine | 0.94 | 0.94 | 0.97 | 1.00 |
| Penguins | 0.96 | 0.99 | 0.99 | 0.99 |

These classes are close to linearly separable, so a linear model does as well or better here.
k-NN comes into its own when the boundary is curved.

## When to use / not use

**Use it** as a quick non-linear baseline with up to a few dozen meaningful features and enough
rows, when the boundary is irregular, or when "here are the similar cases" is a useful
explanation.

**Don't use it** when predictions must be fast or the model small (it stores and searches every
row), with many or noisy features, or when you need calibrated probabilities or coefficients to
explain each feature's effect.

## Common pitfalls

- **Forgetting to scale.** On Wine, accuracy falls from 0.94 to 0.72 without it, and on Penguins
  from 0.99 to 0.80, because the largest-valued feature picks the neighbours. The Lab always
  scales; your own code has to do it too.
- **Choosing $k$ on training scores.** Each training row is its own nearest neighbour, so
  $k = 1$ always looks perfect. Use cross-validation.
- **Trusting the probabilities.** With $k = 5$ they can only be 0, 0.2, 0.4 and so on. They rank
  rows reasonably, but they aren't calibrated, and a probability of 0 hurts log-loss badly.
- **Ignoring imbalance.** With 1% positives, most neighbourhoods have no positive row at all.
  Accuracy will look fine while recall collapses. Try distance weights.
- **Irrelevant columns.** An ID or noise column quietly worsens every distance. Remove it.
- **Treating one-hot columns as ordinary distances.** Any two different categories are the same
  distance apart, and a column with many categories can outweigh the rest.
- **`n_neighbors` larger than the training set.** The run fails, because there aren't that many
  rows to ask.

## Interview questions

1. **What happens during `fit`?** Almost nothing: it validates and stores the rows, and may
   build a search tree. Prediction does the work, which is why it is called lazy.
2. **How does $k$ affect the boundary?** Small $k$: jagged, low bias, high variance. Large $k$:
   smooth, high bias, low variance. At $k = n$ every query gets the majority class.
3. **Why must features be scaled?** Distances sum over features, so those with larger numbers
   dominate regardless of importance.
4. **Why does it degrade with many features?** Distances concentrate: the nearest row is barely
   nearer than the farthest, and neighbourhoods stop being local.
5. **How good can 1-NN get?** With unlimited data its error is at most twice the best possible
   error (Cover & Hart).
6. **Why choose an odd $k$?** To avoid ties between two classes. With more classes ties can still
   happen, and need a rule.
7. **Is it parametric?** No. Its complexity grows with the data, and it stores all of it.
