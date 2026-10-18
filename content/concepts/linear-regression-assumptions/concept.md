---
title: Linear regression assumptions
summary: The conditions under which OLS coefficients and their uncertainty can be trusted.
prerequisites: [residuals]
---

Ordinary least squares always produces a line, but the line is only trustworthy when:

1. **Linearity**: the target is a linear function of the (possibly transformed) features.
2. **Independence**: errors of different rows are unrelated. Repeated measurements or time
   series often violate this.
3. **Homoscedasticity**: errors have roughly constant variance across predictions.
4. **No perfect multicollinearity**: no feature is an exact linear combination of others.
5. **Normal errors**: needed only for confidence intervals and hypothesis tests.

For pure prediction, 1 matters most; for interpreting coefficients, all of them do. Most
violations show up in the residual plot before they show up anywhere else.
