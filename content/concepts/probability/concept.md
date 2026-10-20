---
title: Probability and distributions
summary: Random variables, their mean and variance, and the normal distribution behind noise, errors and sampling.
area: foundations
prerequisites: [numpy-basics]
---

A **random variable** $X$ takes values by chance, following a **distribution**. Two numbers
summarise most distributions:

$$\mathbb{E}[X] = \mu \qquad \operatorname{Var}(X) = \mathbb{E}\big[(X - \mu)^2\big] = \sigma^2$$

The **mean** $\mu$ is the centre; the **variance** $\sigma^2$ is the average squared distance
from it, and the **standard deviation** $\sigma$ is that spread in the data's own units. From
a sample we estimate them with the sample mean $\bar{x}$ and sample variance.

The **normal distribution** $\mathcal{N}(\mu, \sigma^2)$ is the bell curve:

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

About 68% of values fall within $1\sigma$ of the mean, 95% within $2\sigma$, 99.7% within
$3\sigma$. It shows up everywhere because sums of many small independent effects are
approximately normal (the **central limit theorem**). Linear regression assumes its noise is
roughly normal, and minimising MSE is exactly what maximum likelihood gives under that
assumption.

The **law of large numbers** says sample averages settle on the true mean as samples grow,
which is why more data gives steadier estimates.
