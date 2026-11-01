---
title: Gaussian Naive Bayes
summary: Describe each class with one bell curve per feature, then ask which class best explains a new row. Trains in a single pass and needs little data.
difficulty: 2
---

## Intuition

Instead of drawing a boundary between classes, describe each class. Adelie penguins have bills
around 39 mm; Gentoo, around 47 mm. A new penguin with a 46 mm bill looks far more like a Gentoo.

Do that for every feature, multiply the answers together, and weigh in how common each class is.
That's the whole model. The "naive" part is treating features as if they were unrelated within a
class. That's rarely true, yet the model often classifies well anyway.

Move the query to see how each feature's bell curves turn into a probability, then raise the
correlation to see the assumption break.

::figure[naive-bayes-playground]

## How it works

1. Preprocess the features: fill in missing values, standardise numbers, one-hot encode categories.
2. For each class, measure its share of the training rows. This is its **prior**.
3. For each class and feature, measure the mean and variance of that feature within the class.
4. Add a little to every variance, so that no bell curve collapses into a spike.
5. For a new row, score each class: the log of its prior, plus the log height of each of its bell
   curves at the row's values.
6. Predict the class with the highest score. Normalising the scores gives probabilities.

Steps 2 to 4 are all of training. There's no loss, no gradient and no iterations.

## Math

[Bayes' theorem](/concepts/bayes-theorem) with
[conditionally independent](/concepts/conditional-independence) features gives

$$P(c \mid x) \propto P(c) \prod_{j=1}^{d} \mathcal{N}(x_j \mid \mu_{c,j}, \sigma^2_{c,j})$$

and the [maximum-likelihood](/concepts/likelihood) estimates are the sample's own:

$$P(c) = \frac{n_c}{n} \qquad \mu_{c,j} = \frac{1}{n_c}\sum_{i: y_i = c} x_{ij} \qquad \sigma^2_{c,j} = \frac{1}{n_c}\sum_{i: y_i = c} (x_{ij} - \mu_{c,j})^2 + \epsilon$$

where $\epsilon$ is `var_smoothing` times the largest feature variance. A product of many densities
would underflow, so the score is computed as a sum of logs:

$$\text{score}(c) = \log P(c) + \sum_j \left[ -\tfrac{1}{2}\log(2\pi\sigma^2_{c,j}) - \frac{(x_j - \mu_{c,j})^2}{2\sigma^2_{c,j}} \right]$$

Each class has its own variances, so the score is quadratic in $x$, and the decision boundary is
curved. If two classes shared their variances, the squared terms would cancel and their boundary
would be straight.

## Assumptions

- **Features are independent within each class.** Correlated features get counted more than once.
  This mostly harms the probabilities rather than the labels.
- **Each numeric feature is roughly normal within each class.** A long tail or two humps fits a
  single bell curve poorly. Overall normality isn't needed: a feature can look bimodal overall
  because its classes differ, which is what you want.
- **One-hot columns aren't really normal.** They are only ever 0 or 1. The model fits a curve to
  them anyway, which works up to a point (see Common pitfalls).
- **The training rows' class shares match the real world.** They become the priors.

## Hyperparameters

- **`var_smoothing`** is the only one. It's the share of the largest feature variance added to every
  variance. Larger values flatten all the curves, so that no single feature can decide alone.

  Our default is 0.01 rather than scikit-learn's `1e-9`. On the numeric samples, the two score within
  0.002 of each other in cross-validation. On Penguins, 0.01 scores 0.92 against 0.72, for the reason
  explained under Common pitfalls. The results page plots the score per value, and offers the best one
  for your next run.

Unlike k-NN, scaling isn't what makes this model work, since each feature has its own curve. It still
matters a little, because the smoothing is relative to the largest variance.

## When to use / not use

**Use it** as a fast first baseline, when rows are few, when features really are fairly unrelated, or
when training must be cheap. It needs little memory, and new rows can update it without storing old
ones. With few rows it can beat Logistic Regression: it reaches its best sooner, even though that best
is usually lower.

**Don't use it** when you need trustworthy probabilities, when the signal lies in how features combine
(it can't learn XOR), or when you have plenty of data and can afford a stronger model. On the samples
here, cross-validated accuracy is 0.93 against Logistic Regression's 0.98 on Breast Cancer, and 0.92
against 0.99 on Penguins, but 0.95 against 0.96 on Iris.

## Common pitfalls

- **Trusting the probabilities.** On Breast Cancer its average confidence is about 0.99 while its test
  accuracy is about 0.92, and its log-loss is about 0.58 against Logistic Regression's 0.10. The
  results page measures this gap. If you need real probabilities, calibrate them or use another model.
- **A category never seen in a class.** If every Gentoo in the training rows lives on Biscoe, the
  "island is Dream" column is always 0 for Gentoo, so its variance is almost nothing. A Gentoo on
  Dream then scores hugely negative, and one column rules the class out. It works the other way too:
  a narrow curve is very tall, so matching it wins a large bonus. With scikit-learn's default of
  `1e-9`, the penalty is about 500 million. Raise `var_smoothing`, or use a Naive Bayes built for such columns,
  such as Bernoulli.
- **Adding redundant features.** They look harmless, and accuracy may hold up, but each one makes the
  model more overconfident. Consider dropping near-duplicates.
- **Skewed features.** One bell curve fits a long tail badly. A log transform before uploading often
  helps.
- **A sample with the wrong class balance.** If you oversampled a rare class to balance the data, the
  priors are now wrong for real-world use.

## Interview questions

1. **What's naive about it?** It assumes features are independent given the class, so the joint
   likelihood becomes a product of one-dimensional ones.
2. **Why does it work when that's false?** Classification only needs the right class to score highest.
   Dependencies distort the probabilities more than the ranking, and they can cancel out across classes.
3. **Why does it need so little data?** Each feature's parameters are estimated separately, so the
   number of parameters grows linearly with the features: $2d$ per class, with no interactions.
4. **Why add logs instead of multiplying?** A product of many small densities underflows to 0, and logs
   leave the winning class unchanged.
5. **Generative or discriminative?** Generative: it models $P(x \mid c)$ and $P(c)$, then converts with
   Bayes' theorem. Logistic Regression models $P(c \mid x)$ directly.
6. **What is its decision boundary?** Quadratic in general, because each class has its own variances;
   linear if they share them.
7. **What does `var_smoothing` protect against?** A variance near zero, which turns a bell curve into a
   spike so that one value can decide the prediction alone.
8. **Do you need to scale the features?** Not for the likelihood ratios themselves. Only the smoothing
   depends on scale, because it is relative to the largest variance.
