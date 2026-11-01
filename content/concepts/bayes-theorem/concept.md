---
title: Bayes' theorem
summary: Turn "how likely is this evidence if the class is true?" into "how likely is the class, given this evidence?"
area: probabilistic
prerequisites: [probability]
---

A test for a rare disease catches 90% of cases and wrongly flags 5% of healthy people. You test
positive. How worried should you be? Most people say about 90%. The answer here, with 1 person
in 100 sick, is about 15%.

Picture 10,000 people. 100 are sick and 90 of them test positive. Of the 9,900 healthy people,
495 also test positive. So only 90 of the 585 positives are really sick.

Try it. Lower the base rate and watch the false positives swamp the true ones.

::figure[bayes-base-rate]

## The formula

That counting is all the theorem does:

$$P(c \mid x) = \frac{P(x \mid c)\, P(c)}{P(x)}$$

- $P(c)$ is the **prior**: how common the class is before you see anything.
- $P(x \mid c)$ is the **likelihood**: how expected this evidence is if the class is true.
- $P(c \mid x)$ is the **posterior**: what you should believe now.
- $P(x)$ is the **evidence**: how common $x$ is overall. It is $\sum_c P(x \mid c)\,P(c)$, so it
  only makes the posteriors add up to 1.

The likelihood and the posterior run in opposite directions. Confusing them is the mistake in the
opening question: $P(\text{positive} \mid \text{sick}) = 0.9$ doesn't make
$P(\text{sick} \mid \text{positive}) = 0.9$.

## Why classifiers use it

For classification we want $P(c \mid x)$, but it's often easier to describe each class: what do
its rows look like? That's $P(x \mid c)$, and Bayes' theorem converts one into the other. Since
$P(x)$ is the same for every class, picking the most probable class only needs the top of the
fraction:

$$\hat c = \arg\max_c \; P(x \mid c)\, P(c)$$

Models that work this way are called **generative**, because they describe how each class's data
is produced. Models like logistic regression estimate $P(c \mid x)$ directly and are called
**discriminative**.

## Updating, one piece of evidence at a time

Today's posterior can be tomorrow's prior. A second independent positive test, starting from
15%, brings the probability to about 77%. Evidence accumulates. Whether it's fair to multiply
evidence like that is the subject of
[conditional independence](/concepts/conditional-independence).
