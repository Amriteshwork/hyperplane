---
title: Likelihood
summary: Fix the data and vary the parameters. Which settings make what you actually saw least surprising?
area: probabilistic
prerequisites: [probability, logarithms-exponentials]
---

Probability and likelihood use the same formula but ask opposite questions:

- **Probability**: the parameters are fixed. What data might I see?
- **Likelihood**: the data is fixed. Which parameters fit it best?

For one row $x$ and a normal distribution, the likelihood of $\mu, \sigma$ is the height of the
bell curve at $x$:

$$\mathcal{L}(\mu, \sigma \mid x) = p(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

A likelihood is not a probability of the parameters. It doesn't add up to 1 over them, and for a
continuous variable it can be larger than 1, because density isn't probability.

For several independent rows, the densities multiply. Move the curve below to find the settings
that make the data most likely.

::figure[likelihood-explorer]

## Why we add logs

A product of many small numbers underflows: a few thousand densities round to exactly 0 on a
computer. Logs turn it into a sum, and they don't move the maximum because $\log$ only ever
increases:

$$\log \mathcal{L}(\mu, \sigma) = \sum_i \left[ -\tfrac{1}{2}\log(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2} \right]$$

## Maximum likelihood

**Maximum likelihood estimation** picks the parameters with the highest likelihood. For a normal
distribution, setting the derivatives to zero gives something familiar:

$$\hat\mu = \frac{1}{n}\sum_i x_i \qquad \hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \hat\mu)^2$$

The sample mean and variance *are* the best fit, so no optimiser is needed. That's why Gaussian
Naive Bayes trains in one pass.

Many familiar losses are negative log likelihoods in disguise. With $\sigma$ fixed, the only term
depending on $\mu$ is the squared error, so maximising a normal likelihood is minimising
[mean squared error](/concepts/mean-squared-error). For a label with a predicted probability, it
gives [log-loss](/concepts/log-loss).

## Too much likelihood is a warning sign

If all the rows of a class share one value, the best-fitting $\sigma$ is 0. The likelihood at that
value is infinite, and at any other value it's zero. That's overfitting at its most extreme: one
new value rules the class out completely. Models avoid this by putting a floor under $\sigma$, as
Gaussian Naive Bayes's `var_smoothing` does.
