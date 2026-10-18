---
title: Linear Regression
summary: Fit a straight line (or hyperplane) through the data by minimising squared error.
difficulty: 1
---

## Intuition

Imagine plotting house prices against floor area: the points drift upwards in a rough
line. Linear regression draws the single straight line that sits as close as possible to
all of them, then reads predictions off that line. With more features (area, age, distance
to the centre) the line becomes a flat **hyperplane**, but the idea is the same: each feature
pushes the prediction up or down by a fixed amount per unit.

## How it works

1. Start with a model of the form "intercept + weight × feature" for every feature.
2. Measure how wrong it is on the training data with the **mean squared error** (MSE).
3. Choose the weights that make that error as small as possible. For linear regression this
   has an exact solution, the **normal equation**, so there is no trial and error.
4. Predict new rows by plugging their features into the fitted equation.

The same weights can also be found step by step with **gradient descent**, the method used
for models without an exact solution.

## Math

For $n$ examples with $p$ features the model predicts

$$\hat{y}_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip}$$

or, with a column of ones added to the feature matrix $X$, simply $\hat{y} = X\beta$.

Ordinary least squares (OLS) chooses $\beta$ to minimise the squared error

$$L(\beta) = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n}\lVert y - X\beta \rVert^2 .$$

Setting the gradient to zero gives the normal equation and its solution

$$X^\top X \beta = X^\top y \quad\Rightarrow\quad \hat\beta = (X^\top X)^{-1} X^\top y .$$

In practice libraries solve this with a QR or SVD decomposition rather than an explicit
inverse, which is more accurate when features are correlated.

## Assumptions

- **Linearity**: the target changes linearly with each feature (after any transformations
  you apply).
- **Independent errors**: one row's error tells you nothing about another's. Time series often
  break this.
- **Constant variance (homoscedasticity)**: errors are about as large for small predictions as
  for large ones.
- **No perfect multicollinearity**: no feature is an exact combination of others.
- **Roughly normal errors**: only needed for confidence intervals and p-values, not for good
  predictions.

The residual plot in the Lab is the quickest check: a shapeless cloud around zero is good; a
curve or a funnel means an assumption is broken.

## Hyperparameters

Plain OLS has almost nothing to tune, which makes it an honest baseline.

- `fit_intercept`: learn $\beta_0$. Keep it on unless the target is known to be zero when all
  features are zero.
- `positive`: constrain every coefficient to be non-negative, for when a feature can only
  increase the target.

When you need to control overfitting, move to Ridge, Lasso or Elastic Net, which add a
penalty with a strength you tune.

## When to use / not use

**Use it** as the first model on any regression problem: it trains instantly, it is easy to
explain, and every fancier model should beat it to earn its complexity. It shines when
relationships are roughly linear and you care about interpreting the effect of each feature.

**Avoid it** (or transform features first) when relationships are clearly curved, when
features interact strongly, when there are more features than rows, or when the target is a
category. Use logistic regression or another classifier for that.

## Common pitfalls

- **Reading coefficients as causes.** A coefficient is an association holding other features
  fixed, not the effect of an intervention.
- **Correlated features.** Collinear features make coefficients swing wildly between fits even
  when predictions stay stable; Ridge regularisation helps.
- **Outliers.** Squared error lets a few extreme points drag the line; inspect the residuals.
- **Comparing unscaled coefficients.** A coefficient's size depends on the feature's units;
  standardise features before comparing them.
- **Evaluating on training data.** Always report metrics on the held-out test set.

## Interview questions

1. Derive the normal equation from the squared-error loss.
2. Why do libraries use QR/SVD instead of inverting $X^\top X$?
3. What does $R^2$ measure, and how can it be negative on a test set?
4. How do correlated features affect the coefficients and their interpretation?
5. When would you prefer gradient descent over the closed-form solution?
6. How would you detect and handle heteroscedasticity?
