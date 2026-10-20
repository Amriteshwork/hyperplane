---
title: Classification metrics
summary: The confusion matrix and the scores built from it (accuracy, precision, recall, F1), and why accuracy is rarely enough.
area: evaluation
prerequisites: [decision-boundary]
---

Every prediction of a binary classifier falls into one of four cells of the **confusion
matrix**. "Positive" is the class you're looking for:

| | predicted positive | predicted negative |
|---|---|---|
| **actually positive** | true positive (TP) | false negative (FN) |
| **actually negative** | false positive (FP) | true negative (TN) |

The usual scores are ratios of these counts:

- **Accuracy** $= \frac{TP + TN}{n}$: the share right. Misleading when one class dominates.
- **Precision** $= \frac{TP}{TP + FP}$: of the rows called positive, how many were.
- **Recall** $= \frac{TP}{TP + FN}$ (sensitivity, true positive rate): of the real
  positives, how many were found.
- **F1** $= \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$:
  the harmonic mean, high only when both are.

**Precision and recall trade off through the threshold.** Calling more rows positive finds
more positives (recall up) but raises more false alarms (precision down). Which matters
more depends on the cost of each mistake: missing a disease or flagging a healthy patient.

**More than two classes.** Compute each score per class, treating that class as positive
and the rest as negative, then average. **Macro** averaging gives each class an equal vote;
pooling the counts first (**micro**) gives plain accuracy for single-label problems.
