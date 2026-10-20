---
title: Elastic Net
summary: Linear regression with both an L1 and an L2 penalty. It drops unhelpful features like Lasso but keeps correlated ones together like Ridge.
difficulty: 3
---

## Intuition

Ridge and Lasso each have a weak spot. Ridge never drops a feature, so with hundreds of
features you get hundreds of small weights. Lasso drops features eagerly, but given a group
of features that carry the same information (say, height in cm and height in inches), it
keeps one, drops the rest, and may pick a different one next time.

Elastic Net charges for weights **both** ways at once. The L1 part still switches useless
features off. The L2 part makes correlated features share the weight, so the group is kept
or dropped **together**. A single dial, the L1 ratio, slides between the two: all L1 is the
Lasso, nearly all L2 behaves like Ridge.

## How it works

1. Preprocess as usual, and **standardise** the features: both penalties compare weights.
2. Choose the overall strength $\alpha$ and the L1 ratio $\rho$ (between 0 and 1).
3. Minimise *mean squared error + αρ × (sum of absolute weights) + α(1−ρ)/2 × (sum of
   squared weights)* by **coordinate descent**, exactly as for the Lasso. The only change
   is in each one-weight update: the L2 part adds to the denominator, shrinking the weight
   a little more.
4. Predict as linear regression does; zero-weight features drop out.

With two settings, tune them **together**: try a grid of $(\alpha, \rho)$ pairs (for
example $\rho \in \{0.1, 0.5, 0.9, 1\}$ and a log grid of α) and keep the pair with the best
**cross-validated** score.

## Math

scikit-learn's Elastic Net minimises (the intercept $\beta_0$ is not penalised):

$$L(\beta) = \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \alpha\rho \lVert \beta \rVert_1 + \frac{\alpha(1-\rho)}{2}\lVert \beta \rVert_2^2 .$$

With centred data, current residual $r$, $z_j = \tfrac{1}{n}x_j^\top(r + x_j\beta_j)$ and
$c_j = \tfrac{1}{n}\lVert x_j\rVert^2$, the coordinate update is

$$\beta_j \leftarrow \frac{S(z_j,\ \alpha\rho)}{c_j + \alpha(1-\rho)}, \qquad S(z, t) = \operatorname{sign}(z)\max(|z| - t, 0).$$

The soft threshold (L1) decides whether feature $j$ is used; the larger denominator (L2)
shrinks it. With $\rho = 1$ this is exactly the Lasso update.

Every coefficient is zero once $\alpha \ge \alpha_{\max} = \frac{1}{n\rho}\max_j |x_j^\top(y - \bar y)|$.

The squared term makes $L$ **strictly** convex, so the solution is unique even with
perfectly correlated features. For two identical features the L2 part gives them identical
weights (the **grouping effect**), where the Lasso's choice between them is arbitrary. And
unlike the Lasso, which keeps at most $n$ features, Elastic Net can keep more than $n$
when $p > n$.

## Assumptions

The same as linear regression (a roughly linear relationship, independent errors of
similar size), plus:

- **Features on comparable scales**, which standardisation provides.
- **A roughly sparse truth, possibly in groups.** It helps most when some features don't
  matter and those that do come in correlated groups.

## Hyperparameters

- `alpha` (default 0.1): the total penalty strength. It is in the target's units, so a good
  value depends on the data; the training script prints $\alpha_{\max}$ for your L1 ratio,
  and a useful search runs from there down on a log grid. scikit-learn's default is 1.0.
- `l1_ratio` (default 0.5): the share of L1 in the penalty. 1 is the Lasso; small values
  approach Ridge (below about 0.01 the solver becomes unreliable, so use Ridge itself).
  Values close to 1 (0.7–0.95) are a common choice: mostly Lasso, with just enough L2 to
  stabilise correlated groups.
- `fit_intercept`: learn the unpenalised $\beta_0$. Keep it on.
- `max_iter` (default 10 000; scikit-learn's is 1000): the maximum number of
  coordinate-descent sweeps. If a run uses every sweep, raise it.

## When to use / not use

**Use it** when you'd reach for the Lasso but your features are correlated (gene
expression, sensor arrays, overlapping survey questions, one-hot groups), when $p > n$ and
more than $n$ features may matter, or when you want a stabler selection than the Lasso gives.

**Avoid it** when you have few, uncorrelated features: Ridge or Lasso does as well with one
setting to tune instead of two. When most features matter, Ridge is simpler. Like any linear
model, it can't capture strongly non-linear relationships without feature engineering.

## Common pitfalls

- **Forgetting to scale.** Both penalties depend on units without standardisation.
- **Tuning α and ρ one at a time.** They interact: a good α for $\rho = 0.1$ is a poor one
  for $\rho = 0.9$. Search them together.
- **Tuning on the test set.** Two settings make it even easier to overfit the test score.
  Use cross-validation on the training rows, and the test set once at the end.
- **Comparing α across L1 ratios.** The same α means a different amount of L1; compare via
  each ratio's $\alpha_{\max}$.
- **Reading the coefficients as effects.** They are shrunk by both penalties.

## Interview questions

1. Write down the Elastic Net loss. What do the two penalty terms each contribute?
2. Derive the coordinate-descent update and show it reduces to the Lasso's when $\rho = 1$.
3. Why does the Lasso pick one of two identical features arbitrarily, and why does
   Elastic Net give them equal weights?
4. With $p > n$, how many features can the Lasso select? Elastic Net?
5. How would you choose α and the L1 ratio? Why not tune them one at a time?
6. Why does adding the L2 term make the solution unique?
7. When would you choose Ridge or Lasso over Elastic Net despite its flexibility?
