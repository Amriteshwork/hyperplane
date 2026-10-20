---
title: Logarithms and exponentials
summary: eˣ turns sums into products and log turns them back. Probabilities, odds and likelihoods all live on this scale.
area: foundations
prerequisites: [derivatives]
---

The **exponential** $e^x$ (with $e \approx 2.718$) and the natural **logarithm** $\log x$ undo
each other:

$$\log(e^x) = x, \qquad e^{\log y} = y \quad (y > 0).$$

Three properties come up again and again in classification:

- **Products become sums.** $\log(ab) = \log a + \log b$ and $e^{a+b} = e^a e^b$. The
  probability of a whole dataset is a product of one probability per row, far too small to
  store for thousands of rows; its log is a manageable sum. Because log only ever
  increases, maximising the log-likelihood maximises the likelihood.
- **Simple derivatives.** $\frac{d}{dx}e^x = e^x$ and $\frac{d}{dx}\log x = \frac{1}{x}$.
  These keep the gradients of the sigmoid, softmax and log-loss short.
- **Positivity.** $e^x > 0$ for every $x$, so exponentiating any score gives something that
  can be normalised into a probability. That's how softmax works.

In code, exponentials overflow quickly: $e^{710}$ is already too big for a float. The
**log-sum-exp** trick computes $\log\sum_k e^{z_k}$ safely by factoring out the largest term:

$$\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}, \qquad m = \max_k z_k .$$
