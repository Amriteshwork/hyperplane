"""ROC curves and AUC in NumPy: how well scores rank positives above negatives, at every threshold."""

import numpy as np

# @a:reuse
from classification_metrics_numpy import confusion


# @a:roc-curve
def roc_curve(y, scores):
    """(false positive rates, true positive rates), from the strictest threshold to the loosest."""
    y = np.asarray(y)
    positives, negatives = np.sum(y == 1), np.sum(y == 0)
    fpr, tpr = [], []
    for t in np.r_[np.inf, np.unique(scores)[::-1]]:  # every distinct score is a threshold
        tp, fp, _, _ = confusion(y, scores >= t)
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return np.array(fpr), np.array(tpr)


# @a:auc
def auc(fpr, tpr):
    """Area under the curve, by the trapezoid rule."""
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=400) < 0.4).astype(int)
    scores = rng.normal(size=400) + 1.5 * y  # positives score higher, on average

    # @a:curve
    fpr, tpr = roc_curve(y, scores)
    area = auc(fpr, tpr)
    for f, t in list(zip(fpr, tpr))[:: len(fpr) // 6]:
        print(f"FPR {f:.2f} -> TPR {t:.2f}")
    print(f"AUC {area:.3f}")
    assert fpr[0] == tpr[0] == 0 and fpr[-1] == tpr[-1] == 1  # from 'call nothing' to 'call all'
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)

    # @a:ranking
    pos, neg = scores[y == 1], scores[y == 0]
    wins = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
    assert np.isclose(area, wins)  # AUC = P(random positive outscores random negative)

    # @a:invariance
    assert np.isclose(auc(*roc_curve(y, np.exp(3 * scores + 1))), area)  # only the order matters
    assert np.isclose(auc(*roc_curve(y, -scores)), 1 - area)  # flipped scores: flipped AUC
    guess = auc(*roc_curve(y, rng.uniform(size=400)))
    print(f"random scores: AUC {guess:.3f}")
    assert abs(guess - 0.5) < 0.1
