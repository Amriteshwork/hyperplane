---
title: Logistic Regression
summary: A linear model for classification. It scores each row with a weighted sum, then turns the score into class probabilities with a sigmoid or softmax.
difficulty: 2
---

## Intuition

Despite the name, logistic regression is a **classifier**: it predicts a class (benign or
malignant, which penguin species) together with a probability for each class.

It starts like linear regression. Each feature gets a weight, and a row's **score** is the
weighted sum plus an intercept. A score can be any number, so it is squashed into a
probability: with two classes, the S-shaped **sigmoid** maps the score to $P(\text{yes})$;
with more classes, the **softmax** turns one score per class into probabilities that add up
to 1. The predicted class is the most probable one.

Training adjusts the weights until the model gives the true classes high probability, and
is penalised hardest for being confidently wrong. Since the score is a weighted sum, the
classes are separated by a straight line (a plane, in more dimensions): the **decision
boundary**.

## How it works

1. Preprocess: impute missing values, one-hot encode categories and **standardise** the
   numeric features (the penalty compares weights, and the solver converges faster).
2. For each row, compute the score $z = w^\top x + b$ (one score per class with three or more
   classes).
3. Turn scores into probabilities: $\sigma(z)$ for two classes, softmax for more.
4. Measure how well the probabilities fit the true labels with the **log-loss**, add the L2
   penalty $\frac{1}{2C}\lVert w\rVert^2$, and minimise by an iterative solver (L-BFGS). There
   is no closed-form answer as there is for linear regression.
5. Predict the most probable class. With two classes that means $p \ge 0.5$, but you can
   choose a different threshold.

## Math

**Two classes.** scikit-learn's positive class is the last one in sorted order,
`classes_[1]`. With $y_i \in \{0, 1\}$ and $p_i = \sigma(w^\top x_i + b) = \frac{1}{1 + e^{-(w^\top x_i + b)}}$,
it minimises

$$C \sum_{i=1}^{n} -\big[y_i \log p_i + (1-y_i)\log(1-p_i)\big] \;+\; \tfrac12\lVert w\rVert_2^2 .$$

The intercept $b$ is not penalised. Dividing by $Cn$ gives the same minimiser in a friendlier
form: **mean log-loss** plus $\frac{1}{2Cn}\lVert w\rVert^2$.

The model is linear in the **log-odds**:

$$\log\frac{p}{1-p} = w^\top x + b ,$$

so increasing $x_j$ by one unit (one standard deviation, after scaling) multiplies the odds
of the positive class by $e^{w_j}$.

**$K$ classes.** Each class $k$ gets weights $w_k$ and intercept $b_k$, and

$$P(y = k \mid x) = \frac{e^{w_k^\top x + b_k}}{\sum_{j=1}^{K} e^{w_j^\top x + b_j}} ,$$

trained by minimising $C\sum_i -\log P(y_i \mid x_i) + \frac12\sum_k \lVert w_k\rVert^2$.
Adding the same vector to every $w_k$ leaves the probabilities unchanged, so only
differences between classes' coefficients are meaningful.

**The gradient.** For the mean form, $\nabla_w = \frac1n X^\top(p - y) + \frac{1}{Cn}w$: the
same shape as linear regression's, with probabilities in place of predictions. The loss is
convex, so there is a single minimum.

**Class weights.** With weights $s_i$, each row's loss is multiplied by $s_i$; `balanced`
sets $s_i = \frac{n}{K\,n_{y_i}}$ so that every class carries the same total weight.

## Assumptions

- **A linear boundary in the features.** The log-odds must be roughly a weighted sum of the
  (transformed) features. Curved boundaries need engineered features such as squares or
  products.
- **Independent rows.** Repeated measurements of the same patient or customer make the
  model overconfident.
- **Not too much collinearity.** Strongly correlated features make individual coefficients
  unstable (the predictions are fine). The L2 penalty helps.
- **Enough rows of every class.** A rule of thumb is at least 10 rows of the *rarer* class
  per coefficient.

Unlike linear regression, there's no assumption about normal errors or constant variance.

## Hyperparameters

- `C` (default 1.0): the **inverse** of the penalty strength. Small C (0.01) means strong
  shrinkage and cautious probabilities; large C (100) means a fit close to unpenalised
  maximum likelihood. Search on a log scale.
- `class_weight` (default none): `balanced` up-weights rare classes so the model stops
  ignoring them. It raises recall on the rare class at the cost of precision, and inflates
  its predicted probabilities.
- `fit_intercept`: learn the unpenalised $b$. Keep it on.
- `max_iter` (default 5000; scikit-learn's is 100): the maximum L-BFGS iterations. On
  standardised features it usually stops within a few dozen; unscaled features can need
  thousands. If training warns that it did not converge, raise it or lower C.

The penalty is always L2 here. scikit-learn also supports L1 or elastic-net penalties for
logistic regression, with the `saga` solver, which select features as the Lasso does.

## When to use / not use

**Use it** as the first model on almost any classification problem: it is fast, hard to
overfit with sensible C, gives **probabilities** rather than just labels, and its
coefficients explain which features push towards which class. It's the standard in fields
that need interpretable risk scores, such as medicine and credit.

**Avoid it** (or expand the features first) when the classes are separated by curved or
interacting boundaries, like rings, XOR patterns or thresholds on combinations of
features. Trees, kernel methods or neural networks capture those directly. With many raw
features such as pixels, a linear boundary is usually too simple.

## Common pitfalls

- **Judging by accuracy alone.** With a 95/5 class split, predicting the majority class is
  95% accurate. Check precision, recall, F1 and ROC-AUC.
- **Forgetting to scale.** The penalty then shrinks features unevenly, and L-BFGS may not
  converge.
- **Reading coefficients as probabilities.** They are changes in **log-odds**; $e^{w_j}$ is
  an odds ratio.
- **Assuming 0.5 is the right threshold.** Pick the threshold by the cost of each kind of
  mistake.
- **Perfect separation.** If a line separates the training classes perfectly, the
  unpenalised weights grow without bound. The penalty (smaller C) keeps them finite.
- **Comparing coefficients across classes in isolation.** For three or more classes only
  differences between the classes' coefficients are identified.

## Interview questions

1. Why is it called logistic *regression* when it's used for classification?
2. Write down the log-loss. Why not train with mean squared error on the sigmoid output?
3. Derive the gradient of the log-loss for binary logistic regression.
4. How do you interpret a coefficient of 0.7? What is $e^{0.7}$ in this context?
5. What does the decision boundary look like, and how can logistic regression learn a
   non-linear one?
6. What happens to the weights when the training data are perfectly separable? How does
   regularisation fix it?
7. How does the softmax generalise the sigmoid to $K$ classes? Why are the coefficients not
   unique without a penalty?
8. Your positive class is 2% of the data. How do you train and evaluate the model?
