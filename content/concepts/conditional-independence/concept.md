---
title: Conditional independence
summary: Two things can be related overall, yet tell you nothing about each other once you know a third. Naive Bayes assumes this of every feature.
area: probabilistic
prerequisites: [bayes-theorem, likelihood]
---

Taller children tend to know more words. Height doesn't teach vocabulary: both grow with age. Among
children of the same age, knowing one tells you little about the other. Height and vocabulary are
**dependent**, but **conditionally independent given age**:

$$P(a, b \mid c) = P(a \mid c)\, P(b \mid c)$$

Neither kind of independence implies the other, and this page's code shows that with coins: two
tosses of a coin picked at random from a box depend on each other (heads suggests you got the
two-headed one), yet are independent once you know which coin it was.

## The naive assumption

To use [Bayes' theorem](/concepts/bayes-theorem) on rows with $d$ features, we need
$P(x_1, \dots, x_d \mid c)$. That's hopeless to estimate directly: even with 2 values per feature
there are $2^{30}$, over a billion, combinations for 30 features, and most will never appear in
your data.

**Naive Bayes** assumes the features are conditionally independent given the class, so the
impossible joint distribution becomes a product of one-dimensional ones:

$$P(c \mid x) \propto P(c) \prod_{j=1}^{d} P(x_j \mid c)$$

Now there are only $d$ small things to estimate per class. Each needs far fewer rows, and that is
why the model works with so little data.

## What goes wrong: double counting

The assumption is rarely true. Suppose one feature appears twice. Nothing new has been learned, but
its evidence now gets multiplied in twice, and the model grows more certain. Add more copies and it
heads to 100%.

::figure[double-counting]

Correlated features do the same thing to a lesser degree, and real data has plenty of them.

## Why it still classifies well

Double counting exaggerates the evidence, and it often exaggerates it in the direction that was
already winning. So the **ranking** of classes tends to survive even when the **probabilities**
don't. In this page's code, ten copies of a feature leave the accuracy almost unchanged while the
average confidence rises well above it.

So expect good labels and poor probabilities. To use the probabilities as real odds, check them
first with a calibration curve. Assume nothing either way though: where features interact, such as
XOR, each one alone says nothing about the class, and Naive Bayes can't do better than guessing.
