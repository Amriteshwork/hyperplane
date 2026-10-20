---
title: Polynomial Regression
summary: Linear regression on powers and products of the features, so a least-squares fit can follow curves.
difficulty: 2
---

## Intuition

Some relationships bend. Fuel efficiency rises with speed and then falls; crop yield grows
with fertiliser until it starts to hurt. A straight line can't follow that, but a parabola
can. Polynomial regression keeps everything you learned from linear regression and changes
one thing: before fitting, it gives the model extra columns such as $x^2$ and $x_1 x_2$.
The model still adds up weighted columns, but because some columns are curved functions of
the originals, the fitted shape curves too.

## How it works

1. Preprocess as usual: fill in missing values and **standardise** the features.
2. **Expand** the features: add every power and product of them up to a chosen **degree**.
   With two features and degree 2, $[x_1, x_2]$ becomes $[x_1, x_2, x_1^2, x_1 x_2, x_2^2]$.
3. Fit ordinary least squares on the expanded columns, exactly as in linear regression.
4. To predict, expand new rows the same way and take the weighted sum.

The degree is the dial between too simple and too flexible. Pick it by comparing **test**
scores, never training scores.

## Math

For one feature and degree $d$ the model is

$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d .$$

It is non-linear in $x$ but **linear in $\beta$**. Stack the expanded columns into a design
matrix $\Phi$ (for one feature, the Vandermonde matrix with rows $[1, x_i, x_i^2, \dots, x_i^d]$)
and the problem is ordinary least squares:

$$\hat\beta = \arg\min_\beta \lVert y - \Phi\beta \rVert^2 = (\Phi^\top \Phi)^{-1} \Phi^\top y .$$

With $p$ features, all terms up to degree $d$ give

$$\binom{p+d}{d} - 1$$

columns (excluding the intercept). That count grows fast: 10 features at degree 3 already
give 285 columns.

The expected test error splits into $\text{bias}^2 + \text{variance} + \text{noise}$. Raising
the degree lowers bias and raises variance; the best degree balances the two.

## Assumptions

The same as linear regression, applied to the **expanded** features:

- **The right shape**: the true relationship is close to a low-degree polynomial in the
  features over the range you predict on.
- **Independent errors** with **roughly constant variance**.
- **Enough rows per term**: each extra column is another coefficient to estimate.

Unlike plain OLS, perfect collinearity is common and harmless for predictions: with one-hot
features, $x^2 = x$ for a 0/1 column, so the solver picks the minimum-norm solution.

## Hyperparameters

- `degree` (default 2): the highest total power of any term. 1 reproduces linear
  regression; 2 or 3 is enough for most real data; beyond that, overfitting and
  wild extrapolation become likely.
- `interaction_only`: keep products of *different* features ($x_1 x_2$) but no powers
  ($x_1^2$). Useful when features amplify each other but each effect is otherwise linear.
- `fit_intercept`: learn $\beta_0$. Keep it on.

For high degrees or many features, pair the expansion with Ridge or Lasso, which add a
penalty that keeps the many coefficients in check.

## When to use / not use

**Use it** when the residual plot of a linear regression shows a curve, when you suspect
interactions between a handful of features, or when you want a smooth curved fit that is
still fast and has a closed-form solution.

**Avoid it** with many features (the number of terms explodes), with few rows for the
number of terms, and whenever you need predictions **outside the training range**:
polynomials head to $\pm\infty$ beyond the data. Trees, k-nearest neighbours or splines
handle complex shapes more gracefully.

## Common pitfalls

- **Choosing the degree on training data.** Training error always falls as the degree rises.
  Choose by test or cross-validated error.
- **Not scaling first.** Squares and cubes of large numbers dwarf everything else and make
  the solve numerically inaccurate.
- **Extrapolating.** A good fit inside the data says nothing about the curve's behaviour
  outside it.
- **Reading individual coefficients.** $x$, $x^2$ and $x^3$ are highly correlated, so their
  coefficients are unstable and often have large opposite signs. Interpret the curve, not
  the terms.
- **Forgetting the expansion at prediction time.** Save the expansion and the model together
  in one pipeline.

## Interview questions

1. Why is polynomial regression still a *linear* model?
2. How many features does `PolynomialFeatures(degree=d)` produce from $p$ inputs?
3. How would you choose the degree, and why not by training $R^2$?
4. Explain the bias–variance trade-off using polynomial degree as the example.
5. Why should features be standardised before polynomial expansion?
6. Why do polynomial models extrapolate poorly, and what would you use instead?
7. How does Ridge regularisation help a high-degree polynomial?
