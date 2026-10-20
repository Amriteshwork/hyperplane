"""Classification metrics in PyTorch: the confusion matrix, and every score computed from it."""

import torch


# @a:confusion
def confusion(y, y_hat):
    """(tp, fp, fn, tn) for 0/1 (or False/True) labels, where 1 is the positive class."""
    y, y_hat = torch.as_tensor(y).bool(), torch.as_tensor(y_hat).bool()
    return (
        int((y & y_hat).sum()),  # true positives: caught
        int((~y & y_hat).sum()),  # false positives: false alarms
        int((y & ~y_hat).sum()),  # false negatives: missed
        int((~y & ~y_hat).sum()),  # true negatives: correctly left alone
    )


# @a:scores
def precision_recall_f1(y, y_hat):
    """Precision tp/(tp+fp), recall tp/(tp+fn) and their harmonic mean; 0 when undefined."""
    tp, fp, fn, _ = confusion(y, y_hat)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    f64 = torch.float64
    y = (torch.rand(1000) < 0.3).long()  # 30% positives
    p = torch.clamp(0.3 + 0.35 * (2 * y - 1) + 0.2 * torch.randn(1000, dtype=f64), 0, 1)  # a decent model

    # @a:matrix
    y_hat = (p >= 0.5).long()
    tp, fp, fn, tn = confusion(y, y_hat)
    print("          predicted 1  predicted 0")
    print(f"actual 1  {tp:>11}  {fn:>11}")
    print(f"actual 0  {fp:>11}  {tn:>11}")
    assert tp + fp + fn + tn == len(y)

    # @a:metrics
    accuracy = (tp + tn) / len(y)
    precision, recall, f1 = precision_recall_f1(y, y_hat)
    print(f"accuracy {accuracy:.3f}, precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}")
    assert abs(accuracy - (y == y_hat).double().mean().item()) < 1e-12
    assert min(precision, recall) <= f1 <= (precision + recall) / 2  # dragged towards the worse

    # @a:threshold
    trade = [precision_recall_f1(y, p >= t)[:2] for t in (0.3, 0.5, 0.7)]
    for t, (pr, rc) in zip((0.3, 0.5, 0.7), trade):
        print(f"threshold {t}: precision {pr:.3f}, recall {rc:.3f}")
    assert trade[0][1] >= trade[1][1] >= trade[2][1]  # stricter threshold: recall can only fall
    assert trade[0][0] < trade[2][0]  # ...while precision rises

    # @a:multiclass
    yk = torch.multinomial(torch.tensor([0.6, 0.3, 0.1]), 600, replacement=True)  # unequal sizes
    yk_hat = torch.where(torch.rand(600) < 0.8, yk, torch.randint(0, 3, (600,)))  # 80% kept
    per_class = [precision_recall_f1(yk == k, yk_hat == k) for k in range(3)]  # one vs rest
    macro_f1 = sum(f for _, _, f in per_class) / 3
    micro_precision = sum(confusion(yk == k, yk_hat == k)[0] for k in range(3)) / len(yk)
    accuracy_k = (yk == yk_hat).double().mean().item()
    print(f"3 classes: macro F1 {macro_f1:.3f}, accuracy {accuracy_k:.3f}")
    assert abs(micro_precision - accuracy_k) < 1e-12  # pooled over classes: accuracy
