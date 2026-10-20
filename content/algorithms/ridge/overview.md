---
title: Ridge
summary: Linear regression with an L2 penalty that shrinks the coefficients, trading a little bias for much less variance.
difficulty: 2
---

## Intuition

Linear regression picks the weights that fit the training data best, even if that means
huge weights cancelling each other out to chase noise. Ridge adds one rule: **large weights
cost extra**. The model now has to balance two goals, fitting the data and keeping its
weights small, so it settles on a calmer line that fits the training data slightly worse and
new data better.

The idea pays off most when plain OLS is shaky: few rows for the number of features,
features that are strongly correlated, or a flexible model (like a polynomial) that can
overfit.

## How it works

1. Preprocess as usual, and **standardise** the features: the penalty compares weights, so
   the features must be on the same scale.
2. Choose a penalty strength $\alpha$.
3. Find the weights that minimise *squared error + α × (sum of squared weights)*. Like OLS,
   this has an exact solution, with no iterations.
4. Predict exactly as linear regression does: the weighted sum of the features plus the
   intercept.

Try $\alpha$ values a factor of 10 apart and keep the one with the best held-out score,
ideally chosen by **cross-validation** so the test set stays untouched.

## Math

Ridge minimises the penalised least-squares loss (the intercept $\beta_0$ is not penalised):

$$L(\beta) = \lVert y - X\beta \rVert_2^2 + \alpha \lVert \beta \rVert_2^2 = \sum_{i=1}^{n} (y_i - \hat y_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2 .$$

Setting the gradient to zero (with centred $X$ and $y$) gives a modified normal equation:

$$(X^\top X + \alpha I)\,\hat\beta = X^\top y \quad\Rightarrow\quad \hat\beta = (X^\top X + \alpha I)^{-1} X^\top y .$$

Adding $\alpha$ to every eigenvalue of $X^\top X$ makes the matrix invertible even when
features are collinear or $p > n$. In terms of the SVD $X = U \Sigma V^\top$, each component
of the OLS solution is multiplied by $\frac{\sigma_j^2}{\sigma_j^2 + \alpha}$: directions with
little variance in the data (small $\sigma_j$), where OLS is least reliable, are shrunk the
most.

Equivalently, Ridge is OLS under the constraint $\lVert \beta \rVert_2^2 \le t$ for some $t$
that depends on $\alpha$.

## Assumptions

The same as linear regression (a roughly linear relationship, independent errors of
similar size), with two changes:

- **No need for non-collinear features.** Ridge is designed for correlated features and
  works even with more features than rows.
- **Features on comparable scales.** The penalty assumes a unit change in each feature is
  comparable, which is what standardisation provides.

## Hyperparameters

- `alpha` (default 1.0): the penalty strength. Near 0 gives plain OLS; larger values
  shrink all coefficients towards 0 (never exactly to 0). Search it on a log scale (0.01,
  0.1, 1, 10, 100, …) with cross-validation.
- `fit_intercept`: learn the unpenalised $\beta_0$. Keep it on.

## When to use / not use

**Use it** whenever you'd use linear regression but have many features, correlated
features, or few rows per feature. It is rarely worse than OLS when $\alpha$ is tuned, and
often clearly better. It also tames high-degree polynomial features.

**Avoid it** when you need a **sparse** model that names the few features that matter:
Ridge keeps every feature with a small weight, so use Lasso or Elastic Net. And like any
linear model, it can't capture strongly non-linear relationships without feature
engineering.

## Common pitfalls

- **Forgetting to scale.** Without standardisation the penalty depends on units, and
  features measured in large numbers escape it.
- **Tuning α on the test set.** The test score becomes optimistic. Use cross-validation
  on the training data.
- **Searching α on a linear grid.** Its effect spans orders of magnitude; use a log grid.
- **Reading shrunk coefficients as effects.** Ridge coefficients are biased towards zero
  on purpose; they rank features but understate effect sizes.
- **Expecting feature selection.** No coefficient becomes exactly zero.

## Interview questions

1. Derive the Ridge solution and explain why $X^\top X + \alpha I$ is always invertible.
2. Why must features be standardised before Ridge, but not before OLS?
3. Explain Ridge in bias–variance terms. What happens as $\alpha \to 0$ and $\alpha \to \infty$?
4. Why doesn't Ridge set coefficients exactly to zero, while Lasso can?
5. How do you choose $\alpha$, and why not on the test set?
6. How does Ridge behave with two perfectly correlated features?
7. What is the Bayesian interpretation of Ridge (hint: a Gaussian prior on the weights)?
