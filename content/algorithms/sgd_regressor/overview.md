---
title: SGD Regressor
summary: A linear regression trained by stochastic gradient descent, one row at a time, so it scales to data too large for exact solvers.
difficulty: 3
---

## Intuition

Linear Regression and Ridge find their weights by solving equations that involve **every**
row at once. That is exact, but it needs all the data in memory and gets slow with millions
of rows.

The SGD Regressor makes the same kind of prediction, a weighted sum of the features, but
learns the weights the way a person learns from experience: look at one example, see how
wrong the prediction was, nudge the weights a little to be less wrong, move on to the next.
After a few passes over the data (**epochs**), the nudges have added up to weights almost
identical to the exact answer.

This is **stochastic gradient descent**, the optimiser behind almost all of deep learning,
in its simplest setting: a model you already understand, so you can watch the optimiser
itself.

## How it works

1. Preprocess and **standardise** the features. Every update moves each weight in
   proportion to its feature's value, so unscaled features make one learning rate too big
   for some and too small for others. Standardise the **target** too, so the step sizes and
   the stopping tolerance don't depend on its units; predictions are converted back.
2. Start with all weights at 0.
3. For each epoch: shuffle the training rows; for each row, compute its prediction error and
   step the weights against that row's gradient, scaled by the **learning rate**.
4. Shrink the learning rate as training goes on (the schedule), so the noisy steps settle.
5. Stop when the average training loss hasn't improved by `tol` for 5 epochs in a row, or
   after `max_iter` epochs.
6. Predict with the weighted sum, exactly as linear regression does.

## Math

The model is $\hat y = w^\top x + b$. SGD minimises the mean loss plus a penalty:

$$J(w, b) = \frac1n\sum_{i=1}^{n} \ell\big(w^\top x_i + b - y_i\big) + \alpha\,R(w).$$

With the default squared error $\ell(e) = \frac12 e^2$ and $R(w) = \frac12\lVert w\rVert_2^2$,
this is Ridge's objective with Ridge's $\alpha$ equal to $\alpha n$: the same minimum,
reached by a different route.

The gradient of $J$ is an average over rows. SGD replaces it by **one** row's gradient, for a
row $i$ picked in shuffled order:

$$w \leftarrow w - \eta_t\big(\ell'(e_i)\,x_i + \alpha w\big), \qquad b \leftarrow b - \eta_t\,\ell'(e_i), \qquad e_i = w^\top x_i + b - y_i .$$

For squared error $\ell'(e) = e$: each row pulls the weights in proportion to its error and
its features. The $\alpha w$ term shrinks every weight slightly each step ("weight decay").

That one-row gradient is **unbiased**: averaged over a random row it equals the full
gradient. Its noise is why the learning rate must decay. The default schedule is
$\eta_t = \eta_0 / t^{0.25}$, with $t$ counting single-row updates.

**Other losses.** Huber, $\frac12 e^2$ for $|e| \le \varepsilon$ and
$\varepsilon|e| - \frac12\varepsilon^2$ beyond, caps each row's pull at $\varepsilon$, so
outliers matter less. ε-insensitive, $\max(0, |e| - \varepsilon)$, ignores errors smaller
than $\varepsilon$, which is the loss of support vector regression.

## Assumptions

The same as Linear Regression's, since the model is the same:

- **A roughly linear relationship** between the (transformed) features and the target.
- **Independent rows**, and errors without strong patterns left in the residuals.

And one about the optimiser:

- **Features on a common scale.** Not optional here: unscaled features make training crawl
  or diverge.

## Hyperparameters

- `loss` (default `squared_error`): what an error costs; `huber` resists outliers,
  `epsilon_insensitive` ignores small errors.
- `learning_rate` (default `invscaling`): the schedule. `constant` keeps $\eta_0$; `adaptive`
  keeps $\eta_0$ until the loss stalls, then divides it by 5; `optimal` is
  $\frac{1}{\alpha(t_0 + t)}$, chosen from α.
- `eta0` (default 0.001; scikit-learn's is 0.01): the starting learning rate. The most
  important setting: too large and the loss explodes, too small and it needs many epochs.
  The lower default keeps rows with extreme feature values from derailing training.
- `alpha` (default 0.0001): the penalty strength, per row.
- `penalty` (default `l2`), `l1_ratio`: `l1` and `elasticnet` give Lasso- and
  Elastic-Net-style sparsity.
- `epsilon` (default 0.1): the switch point for `huber` and the dead zone for
  `epsilon_insensitive`, in standard deviations of the (standardised) target.
- `max_iter` (default 1000), `tol` (default 0.00001; scikit-learn's is 0.001): the maximum
  number of epochs and the stopping tolerance. The small default step needs the small
  tolerance, or training stops before reaching the minimum.
- `early_stopping` (default off): stop on a held-out 10% of the training rows instead.
- `random_state` (default 0): the shuffle seed. A different seed gives slightly different
  weights.

## When to use / not use

**Use it** when the data are large (hundreds of thousands of rows or more), arrive as a
stream, or don't fit in memory: each update needs only one row, and scikit-learn's
`partial_fit` can keep learning from new batches. Use it, too, to understand the optimiser
that trains neural networks, on a model whose exact answer you can check.

**Avoid it** on small and medium data, where Linear Regression or Ridge are exact, instant
and need no learning rate. Like any linear model, it can't capture curved relationships
without engineered features.

## Common pitfalls

- **Unscaled features.** The classic way to see the loss explode, or nothing happen.
- **A learning rate that's too large.** The coefficients come out huge and R² hugely
  negative. Divide `eta0` by 10.
- **Too few epochs.** A `ConvergenceWarning` means it hit `max_iter` first; the weights may
  still be far from the minimum.
- **Expecting identical results every run.** The weights depend on the shuffle order; fix
  `random_state` to reproduce a run.
- **Outliers in the target.** With squared error, one extreme row gives huge steps. Try
  `huber`.
- **An unscaled target.** With a target in the thousands, steps are huge and the loss is
  never within `tol` of its best, so training runs to `max_iter`. Standardise the target
  (this pack does, with `TransformedTargetRegressor`).
- **Extreme rows.** One row whose scaled features are far out (100 standard deviations)
  can wreck the weights in one large step. Lower `eta0`, or clip or transform the feature.

## Interview questions

1. What's the difference between batch, mini-batch and stochastic gradient descent?
2. Why is a single row's gradient a useful estimate of the full gradient? In what sense is it
   unbiased?
3. Why does SGD need a decaying learning rate to converge, when gradient descent doesn't?
4. Why is feature scaling more important for SGD than for the normal equation?
5. What is an epoch? Why reshuffle the data every epoch?
6. With squared error and an L2 penalty, which model does SGDRegressor converge to? How do the
   α's compare?
7. When would you choose SGD over a closed-form solution?
8. How does the Huber loss change each row's update, and why does that help with outliers?
