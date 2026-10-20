---
title: Sigmoid (logistic) function
summary: Squash any real number into a probability between 0 and 1. Its inverse, the log-odds, is what logistic regression models linearly.
area: foundations
prerequisites: [logarithms-exponentials, probability]
---

The **sigmoid** (or logistic function) maps any real score $z$ to a number between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}} .$$

$\sigma(0) = 0.5$; large positive scores approach 1 and large negative ones 0. It is
symmetric, $\sigma(-z) = 1 - \sigma(z)$.

**Its inverse is the log-odds.** If $p = \sigma(z)$, then

$$z = \log\frac{p}{1 - p} .$$

The **odds** $p/(1-p)$ compare "yes" to "no" (0.8 means 4 to 1), and their log can be any
real number. Logistic regression is a **linear model of the log-odds**:
$\log\frac{p}{1-p} = w^\top x + b$. A coefficient $w_j$ is the change in log-odds per unit of
$x_j$: the odds are multiplied by $e^{w_j}$.

**Its derivative is built from itself:**

$$\sigma'(z) = \sigma(z)\,(1 - \sigma(z)) \le \tfrac14 .$$

That is what makes the log-loss gradient so simple, and the bound $\tfrac14$ gives
gradient descent a safe step size.

Computed naively, $e^{-z}$ overflows for very negative $z$. Writing
$\sigma(z) = e^{-\log(1 + e^{-z})}$ with a stable `logaddexp` avoids it.
