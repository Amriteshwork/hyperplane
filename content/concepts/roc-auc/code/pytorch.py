"""ROC curves and AUC in PyTorch: how well scores rank positives above negatives, at every threshold."""

import torch

# @a:reuse
from classification_metrics_pytorch import confusion


# @a:roc-curve
def roc_curve(y, scores):
    """(false positive rates, true positive rates), from the strictest threshold to the loosest."""
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    thresholds = torch.cat([torch.tensor([float("inf")], dtype=scores.dtype),
                            torch.unique(scores).flip(0)])  # every distinct score is a threshold
    fpr, tpr = [], []
    for t in thresholds:
        tp, fp, _, _ = confusion(y, scores >= t)
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return torch.tensor(fpr, dtype=torch.float64), torch.tensor(tpr, dtype=torch.float64)


# @a:auc
def auc(fpr, tpr):
    """Area under the curve, by the trapezoid rule."""
    return float(torch.trapezoid(tpr, fpr))


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    f64 = torch.float64
    y = (torch.rand(400) < 0.4).long()
    scores = torch.randn(400, dtype=f64) + 1.5 * y  # positives score higher, on average

    # @a:curve
    fpr, tpr = roc_curve(y, scores)
    area = auc(fpr, tpr)
    for f, t in list(zip(fpr.tolist(), tpr.tolist()))[:: len(fpr) // 6]:
        print(f"FPR {f:.2f} -> TPR {t:.2f}")
    print(f"AUC {area:.3f}")
    assert fpr[0] == tpr[0] == 0 and fpr[-1] == tpr[-1] == 1  # from 'call nothing' to 'call all'
    assert torch.all(fpr.diff() >= 0) and torch.all(tpr.diff() >= 0)

    # @a:ranking
    pos, neg = scores[y == 1], scores[y == 0]
    wins = (pos[:, None] > neg[None, :]).double().mean() + 0.5 * (pos[:, None] == neg[None, :]).double().mean()
    assert abs(area - wins.item()) < 1e-9  # AUC = P(random positive outscores random negative)

    # @a:invariance
    assert abs(auc(*roc_curve(y, torch.exp(3 * scores + 1))) - area) < 1e-9  # only the order matters
    assert abs(auc(*roc_curve(y, -scores)) - (1 - area)) < 1e-9  # flipped scores: flipped AUC
    guess = auc(*roc_curve(y, torch.rand(400, dtype=f64)))
    print(f"random scores: AUC {guess:.3f}")
    assert abs(guess - 0.5) < 0.1
