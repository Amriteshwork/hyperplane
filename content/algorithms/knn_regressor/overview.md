---
title: KNN Regressor
summary: Predict a number by averaging the targets of the most similar training rows. No formula and no training, just a search.
difficulty: 1
---

## Intuition

To price a house, look up the five most similar houses that have sold and average their
prices. That is the whole algorithm. There is no line to fit and no weights to learn: the
model is the training data, and similarity does all the work.

Since nothing assumes a shape, the prediction follows whatever the data does: curves,
plateaus, interactions. What it assumes instead is that **rows with similar features have
similar targets**.

Slide $k$ and watch the fit. At $k = 1$ it chases every point; at large $k$ it flattens out.

::figure[knn-regression]

## How it works

1. Preprocess and **standardise** the features. This step is essential here.
2. "Train" by storing the rows. Nothing else is computed, apart from a search tree to speed
   up later lookups.
3. For each query, apply the same preprocessing and measure the distance to the stored rows.
4. Take the $k$ nearest.
5. Predict their average target, or a distance-weighted average.

## Math

With $N_k(x)$ the $k$ training rows closest to $x$:

$$\hat y(x) = \frac{1}{k}\sum_{i \in N_k(x)} y_i \qquad \text{or, with distance weights,} \qquad \hat y(x) = \frac{\sum_{i \in N_k(x)} y_i / d(x, x_i)}{\sum_{i \in N_k(x)} 1 / d(x, x_i)}.$$

The default distance is Euclidean, $d(x, z) = \lVert x - z \rVert_2$. No loss is minimised during
training. The average is, though, the constant that minimises squared error over the
neighbours, so k-NN is least squares done locally, afresh for each query.

$k$ sets the bias–variance trade-off. For a noisy target with noise variance $\sigma^2$, the
variance of the average is $\sigma^2 / k$: larger $k$ steadies the prediction, while reaching
for rows farther away adds bias.

A brute-force search costs $O(nd)$ per query, with $n$ rows and $d$ features.

## Assumptions

- **Smoothness:** nearby rows have similar targets. If they don't, no $k$ will help.
- **Meaningful distance:** every feature, once scaled, matters about equally. An irrelevant
  feature adds noise to every distance.
- **Enough rows for the dimension:** the rows must fill the space densely enough to find
  neighbours that are actually close (see the curse of dimensionality).
- **New rows look like old ones:** predictions are averages of training targets, so they can
  never leave that range.

## Hyperparameters

| Setting | Effect |
|---|---|
| `n_neighbors` ($k$) | Small: wiggly, low bias, high variance. Large: smooth, high bias, low variance. Start at 5, then use the cross-validated plot in the results. |
| `weights` | `uniform` counts every neighbour alike. `distance` favours the closest, which reduces the harm of a large $k$. |
| `metric` | `euclidean` (the default), `manhattan` (less swayed by one large difference) or `cosine` (direction, not length). |

Measured on the sample datasets (test $R^2$, standardised features):

| Dataset | $k=1$ | $k=5$ | $k=15$ | Linear Regression |
|---|---|---|---|---|
| California Housing | 0.49 | 0.67 | 0.68 | 0.58 |
| Diabetes | 0.07 | 0.43 | 0.42 | 0.45 |

California's relationships are local (location matters), so k-NN beats a straight line. On
Diabetes, which is nearly linear with only 442 rows, it doesn't.

## When to use / not use

**Use it** as a quick non-linear baseline with up to a few dozen meaningful features and enough
rows, when the relationship is local, or when "here are the similar cases" is a useful
explanation.

**Don't use it** when predictions must be fast or the model small (it stores and searches every
row), with many or noisy features, when you need to predict beyond the range of the training
targets, or when you need coefficients to explain each feature's effect.

## Common pitfalls

- **Forgetting to scale.** On California Housing, $R^2$ drops from 0.67 to 0.15 without it,
  because the largest-valued feature picks the neighbours. The Lab always scales; your own
  code has to do it too.
- **Choosing $k$ on training scores.** Each training row is its own nearest neighbour, so
  $k = 1$ always looks perfect. Use cross-validation.
- **Fitting the scaler on all the data.** Keep it in the pipeline, so it learns from the
  training rows only.
- **Irrelevant columns.** An ID or noise column quietly worsens every distance. Remove it.
- **Treating one-hot columns as ordinary distances.** Any two different categories are the same
  distance apart, and a column with many categories can outweigh the rest.
- **Expecting extrapolation.** A house larger than any in the training data gets the average of
  the largest ones.
- **`n_neighbors` larger than the training set.** The run fails, because there aren't that many
  rows to ask.

## Interview questions

1. **What happens during `fit`?** Almost nothing: it validates and stores the rows, and may
   build a search tree. Prediction does the work, which is why it is called lazy.
2. **How does $k$ affect bias and variance?** Small $k$: low bias, high variance. Large $k$: high
   bias, low variance. At $k = n$ it predicts the overall mean.
3. **Why must features be scaled?** Distances sum over features, so those with larger numbers
   dominate regardless of importance.
4. **Why does it degrade with many features?** Distances concentrate: the nearest row is barely
   nearer than the farthest, and neighbourhoods stop being local.
5. **Can it extrapolate?** No. Every prediction is an average of training targets.
6. **What does a prediction cost?** $O(nd)$ by brute force. Trees help in low dimensions and lose
   their advantage as $d$ grows.
7. **How does it relate to least squares?** The mean minimises squared error, so it fits a
   constant to each query's neighbourhood.
