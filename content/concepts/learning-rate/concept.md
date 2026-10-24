---
title: Learning rate
summary: The step size of gradient descent. Too small and training crawls; too large and the loss explodes.
area: optimisation
prerequisites: [gradient-descent]
---

Gradient descent moves the parameters by $\eta \nabla L$. The **learning rate** $\eta$ is the
one number that decides whether that works:

- **too small**: every step is safe but tiny, and training needs thousands of them;
- **just right**: the loss drops quickly and levels off at the minimum;
- **too large**: each step overshoots the minimum by more than it started from, and the loss
  grows without bound.

The cliff is not arbitrary. On a quadratic loss such as MSE, gradient descent is stable
exactly when $\eta < 2/\lambda_{\max}$, where $\lambda_{\max}$ is the loss's largest
curvature (for MSE, the largest eigenvalue of $\frac{2}{n}X^\top X$). Features on very
different scales make $\lambda_{\max}$ huge in one direction, which forces a tiny $\eta$ for
all of them: the main reason to standardise before gradient descent.

With **stochastic** gradients the steps are noisy, and a constant $\eta$ keeps jittering
around the minimum forever. A **schedule** shrinks it over time, e.g.
$\eta_t = \eta_0 / t^{p}$ or $\eta_t = \frac{1}{\alpha(t_0 + t)}$: big steps to get close,
then small ones to settle.
