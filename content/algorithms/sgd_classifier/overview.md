---
title: SGD Classifier
summary: A linear classifier (logistic regression or a linear SVM, depending on its loss) trained by stochastic gradient descent, one row at a time.
difficulty: 3
---

## Intuition

Logistic Regression scores each row with a weighted sum and finds its weights with a solver
that looks at all the data in every iteration. The SGD Classifier keeps the weighted sum but
learns it the cheap way: look at one row, see how wrong the score was, nudge the weights,
move on. A few passes (**epochs**) over the shuffled data are usually enough.

The **loss** decides which classifier you are training:

- `log_loss`: logistic regression, with probabilities;
- `hinge`: a linear **support vector machine**, which only cares about rows close to or on
  the wrong side of the boundary;
- `modified_huber`: a smoothed hinge that also gives probabilities and tolerates outliers.

All of them separate the classes with a straight boundary. The point of SGD is scale: each
update needs one row, so it handles millions of rows, or a stream of them.

## How it works

1. Preprocess and **standardise** the features, so one learning rate suits every weight.
2. With $K \ge 3$ classes, set up $K$ yes/no problems, one per class ("this class or not?"):
   **one-vs-rest**. With two classes there is one problem, for `classes_[1]`.
3. For each problem, start from zero weights. Each epoch: shuffle the rows, and for each row
   step the weights against the gradient of that row's loss.
4. Shrink the learning rate as training goes on; stop when the average loss stalls.
5. Predict the class whose score is highest (two classes: a positive score means
   `classes_[1]`).

## Math

With labels written as $y_i \in \{-1, +1\}$ and score $z_i = w^\top x_i + b$, SGD minimises

$$J(w, b) = \frac1n\sum_{i=1}^{n} \ell(y_i z_i) + \tfrac{\alpha}{2}\lVert w\rVert_2^2 .$$

$y_i z_i > 0$ means the row is on the right side of the boundary $w^\top x + b = 0$. The
losses:

- **log-loss** $\ell(m) = \log(1 + e^{-m})$, so $P(y = +1) = \sigma(z)$. This is Logistic
  Regression's objective with $C = 1/(\alpha n)$;
- **hinge** $\ell(m) = \max(0, 1 - m)$: zero once a row is beyond the **margin** $m \ge 1$;
- **modified Huber**: $\max(0, 1-m)^2$ for $m \ge -1$, and $-4m$ below.

The update for row $i$ uses only that row's gradient:

$$w \leftarrow w - \eta_t\big(\ell'(y_i z_i)\,y_i x_i + \alpha w\big), \qquad b \leftarrow b - \eta_t\,\ell'(y_i z_i)\,y_i .$$

With the hinge loss, rows beyond the margin have $\ell' = 0$ and leave the weights alone
(except the penalty's shrinkage): only the hard rows teach the model.

The default schedule `optimal` is $\eta_t = \frac{1}{\alpha(t_0 + t)}$: a $1/t$ decay, the
fastest that theory allows on this kind of convex loss, with $t_0$ chosen so the first
steps are about the size of a typical weight.

**Probabilities with $K$ classes.** Each one-vs-rest model gives its own
$\sigma(z_k)$; scikit-learn divides them by their sum so they add up to 1. That is not the
softmax of multinomial Logistic Regression, so the probabilities differ a little.

## Assumptions

- **A linear boundary in the features**, as for Logistic Regression.
- **Independent rows.**
- **Features on a common scale**, for the optimiser.
- **Enough rows of every class**: each one-vs-rest problem is imbalanced (one class against
  all the others), more so with many classes.

## Hyperparameters

- `loss` (default `log_loss`; scikit-learn's own default is `hinge`): which model to train.
  Only `log_loss` and `modified_huber` give probabilities, and so log-loss and ROC-AUC.
- `learning_rate` (default `optimal`): the schedule. `invscaling` is $\eta_0/\sqrt t$,
  `constant` keeps $\eta_0$, `adaptive` divides it by 5 whenever the loss stalls.
- `eta0` (default 0.01): the starting learning rate for every schedule except `optimal`.
- `alpha` (default 0.0001): the penalty strength, per row; it also scales `optimal`'s steps.
- `class_weight` (default none): `balanced` up-weights rare classes.
- `penalty` (default `l2`), `l1_ratio`: `l1`/`elasticnet` for sparse weights.
- `max_iter` (default 1000), `tol` (default 0.001), `early_stopping` (default off): when to
  stop.
- `random_state` (default 0): the shuffle seed.

## When to use / not use

**Use it** for classification on large or streaming data (scikit-learn's `partial_fit`
updates a model with each new batch), or with very many features, such as text, where its
updates stay cheap. Switch between logistic regression and a linear SVM by changing one
setting. And use it to see how the optimiser behind neural networks behaves.

**Avoid it** on small data: Logistic Regression's solver finds the exact minimum, needs no
learning rate and gives the same answer every time. For boundaries that aren't straight, use
a tree, kernel or neural model.

## Common pitfalls

- **Forgetting to scale.** Training becomes unstable or very slow.
- **Expecting probabilities from the hinge loss.** It has none; `predict_proba` doesn't
  exist, and neither do log-loss or ROC-AUC in the results.
- **Reading noise as signal.** Rerun with another `random_state`: if accuracy or weights
  move a lot, the data are too small for SGD to settle.
- **Too few epochs or too large a step.** Accuracy that jumps between runs or epochs means
  the learning rate is too large for this data.
- **Judging by accuracy alone** on imbalanced classes. Check recall, F1 and ROC-AUC, and try
  `class_weight="balanced"`.
- **Comparing its probabilities with Logistic Regression's for 3+ classes.** One-vs-rest
  normalised sigmoids aren't the softmax.

## Interview questions

1. How do the log-loss and hinge loss differ, and which models do they give?
2. Why do rows beyond the margin not move the weights under the hinge loss?
3. With log-loss and an L2 penalty, how is SGDClassifier's α related to LogisticRegression's
   C?
4. How does one-vs-rest handle three or more classes? What are its weaknesses?
5. Why does the learning rate need to decay for SGD to converge?
6. Why might two runs of SGDClassifier on the same data give different weights?
7. When would you choose SGD over LogisticRegression's default solver?
8. Why is the hinge loss model unable to give probabilities?
