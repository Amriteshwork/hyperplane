---
title: Lasso
summary: Linear regression with an L1 penalty that sets some coefficients exactly to zero, so it fits and selects features at once.
difficulty: 2
---

## Intuition

Ridge makes large weights cost extra, so every weight shrinks a little. Lasso charges for
weights differently: each one costs its **absolute size**, so a small weight is just as
expensive per unit as a large one. For a feature that only helps a little, the cheapest
option is to give it no weight at all. Lasso does exactly that: it switches the least
useful features **off**, and fits the rest.

The result is a **sparse** model, a weighted sum of only a few features. It is easier to
read, cheaper to use, and often generalises as well as the full model, because the dropped
features were mostly fitting noise.

## How it works

1. Preprocess as usual, and **standardise** the features: the penalty compares weights, so
   which features survive must not depend on their units.
2. Choose a penalty strength $\alpha$.
3. Minimise *mean squared error + α × (sum of absolute weights)*. There's no exact formula
   this time, so the weights are found by **coordinate descent**: update one weight at a
   time to its best value given the others, sweep over all of them, and repeat until they
   stop changing. Each update can land exactly on 0.
4. Predict exactly as linear regression does, with the zero-weight features dropping out.

Larger $\alpha$ zeroes more features. Above $\alpha_{\max}$ every weight is zero; the useful
range is below it, searched on a log scale and ideally chosen by **cross-validation**.

## Math

scikit-learn's Lasso minimises (the intercept $\beta_0$ is not penalised):

$$L(\beta) = \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \alpha \lVert \beta \rVert_1, \qquad \lVert \beta \rVert_1 = \sum_{j=1}^{p} |\beta_j| .$$

$|\beta_j|$ has a corner at 0, so $L$ has no gradient there and no closed form. But along
one coordinate, with the others fixed, the minimiser is explicit. With centred data and
$r = y - X\beta$ the current residual,

$$\beta_j \leftarrow \frac{S(\rho_j, \alpha)}{c_j}, \qquad \rho_j = \tfrac{1}{n} x_j^\top (r + x_j \beta_j), \quad c_j = \tfrac{1}{n}\lVert x_j \rVert_2^2,$$

where $S(z, t) = \operatorname{sign}(z)\max(|z| - t, 0)$ is the **soft-threshold** operator.
Whenever $|\rho_j| \le \alpha$, feature $j$'s coefficient is set to exactly 0. Cycling these
updates converges to the global minimum because $L$ is convex.

Starting from $\beta = 0$, every coefficient stays at zero once

$$\alpha \ge \alpha_{\max} = \tfrac{1}{n}\max_j \lvert x_j^\top (y - \bar y) \rvert .$$

Geometrically, Lasso is least squares under the constraint $\lVert\beta\rVert_1 \le t$: a
diamond whose corners lie on the axes. The squared-error contours usually first touch the
diamond at a corner, where some coefficients are 0. Ridge's constraint is a circle, which
has no corners. In Bayesian terms, Lasso is the most likely $\beta$ under a Laplace prior,
whose sharp peak at zero favours exact zeros.

## Assumptions

The same as linear regression (a roughly linear relationship, independent errors of
similar size), plus:

- **Features on comparable scales**, which standardisation provides.
- **The truth is (roughly) sparse.** Lasso helps most when only some features matter.
  When most of them contribute a little, Ridge usually predicts better.
- **Correlated features are interchangeable.** From a group of strongly correlated
  features Lasso tends to keep one, and which one can change between samples.

## Hyperparameters

- `alpha` (default 0.1): the penalty strength. Larger values zero more coefficients. It is
  in the target's units, so a good value depends on the data: the training script prints
  $\alpha_{\max}$, and a useful search runs from there down on a log grid, chosen by
  cross-validation. scikit-learn's own default is 1.0.
- `fit_intercept`: learn the unpenalised $\beta_0$. Keep it on.
- `max_iter` (default 10 000; scikit-learn's is 1000): the maximum number of
  coordinate-descent sweeps. One-hot encoded features slow convergence, and so does a very
  small $\alpha$; if a run uses every sweep, raise it.

## When to use / not use

**Use it** when you have many features and suspect only a few matter, when you want a
short, readable model, or when collecting features is costly and you want to know which
ones you could stop collecting. It also tames wide data with few rows.

**Avoid it** when most features contribute (Ridge shrinks without discarding and usually
predicts better), when features come in correlated groups that you want kept together
(Elastic Net combines both penalties for this), or when $p > n$ and more than $n$ features
matter: Lasso keeps at most $n$. Like any linear model, it can't capture strongly
non-linear relationships without feature engineering.

## Common pitfalls

- **Forgetting to scale.** Without standardisation, features in large units need small
  weights, pay little penalty and survive; selection then reflects units, not usefulness.
- **Reusing Ridge's α.** The two penalties are measured differently, and scikit-learn's
  Lasso divides the squared error by $2n$. Search Lasso's α afresh, below $\alpha_{\max}$.
- **Treating the selection as the truth.** With correlated features the kept set is one
  of several near-equivalent choices. Refit on bootstrap samples to see how stable it is.
- **Reading shrunk coefficients as effects.** Even the kept coefficients are biased towards
  zero. Refit plain OLS on the selected features (the "relaxed lasso") for unbiased sizes.
- **Ignoring convergence warnings.** A fit that hit `max_iter` isn't the Lasso solution.

## Interview questions

1. Why does the L1 penalty produce exact zeros while L2 doesn't? Explain with the geometry
   and with the soft-threshold update.
2. Why is there no closed-form Lasso solution, and how does coordinate descent get around
   that?
3. Derive $\alpha_{\max}$, the smallest penalty at which all coefficients are zero.
4. How does Lasso behave with two perfectly correlated features? How does Elastic Net
   fix it?
5. With $p > n$, at most how many features can Lasso select, and why?
6. What prior on the weights corresponds to Lasso? To Ridge?
7. Why must features be standardised before Lasso, and why shouldn't the intercept be
   penalised?
