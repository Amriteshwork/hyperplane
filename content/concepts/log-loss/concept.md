---
title: Log-loss (cross-entropy)
summary: Score predicted probabilities by −log of the probability given to the true class. Minimising it is maximum likelihood.
area: optimisation
prerequisites: [sigmoid, loss-function, probability]
---

A classifier that outputs probabilities should be judged by them, not just by its final
label. **Log-loss** (also called cross-entropy or negative log-likelihood) charges each row
$-\log$ of the probability the model gave to its true class. For two classes, with
$p_i = P(y_i = 1)$:

$$\text{log-loss} = -\frac{1}{n}\sum_{i=1}^{n} \big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big] .$$

For $K$ classes it is $-\frac{1}{n}\sum_i \log p_i(y_i)$, the same idea.

**Why log?**

- **It is maximum likelihood.** The probability of all the observed labels is
  $\prod_i p_i(y_i)$. Taking $-\frac1n\log$ turns that product into the log-loss, so the
  weights that minimise log-loss are those under which the data was most probable.
- **Confident mistakes are expensive.** Predicting 0.01 for a true 1 costs
  $-\log 0.01 \approx 4.6$; squared error would charge at most 1. The model learns never to
  be sure and wrong.
- **A clean gradient.** With $p = \sigma(Xw + b)$, the derivative of the log cancels the
  sigmoid's, leaving

  $$\nabla_w = \frac{1}{n}X^\top(p - y) ,$$

  the same form as linear regression's MSE gradient. The loss is also **convex** in $w$,
  so gradient descent finds the global minimum. (MSE on sigmoid outputs is neither.)

Lower is better; 0 is perfect. Always guessing the class proportions gives a baseline
worth comparing against.
