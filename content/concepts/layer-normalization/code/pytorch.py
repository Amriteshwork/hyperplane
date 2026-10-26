"""Layer normalization in PyTorch: per-token standardisation, checked against nn.LayerNorm."""

import torch


# @a:layer-norm
def layer_norm(X, gamma, beta, eps=1e-5):
    """Each row (token) to mean 0 and variance 1 across its features, then × γ + β per feature."""
    centred = X - X.mean(dim=-1, keepdim=True)
    var = (centred**2).mean(dim=-1, keepdim=True)  # divide by d, not d − 1
    return gamma * centred / torch.sqrt(var + eps) + beta


# @a:demo
if __name__ == "__main__":
    # @a:builtin
    torch.manual_seed(0)
    B, T, d = 2, 5, 8
    X = torch.randn(B, T, d) * 10 + 3
    norm = torch.nn.LayerNorm(d)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 2.0)  # pretend γ and β have been trained
        norm.bias.normal_()
        assert torch.allclose(layer_norm(X, norm.weight, norm.bias), norm(X), atol=1e-5)
    print("matches nn.LayerNorm, learned γ and β included")

    # @a:batch-norm
    sentence = torch.randn(T, d)
    batch_a = torch.stack([sentence, torch.randn(T, d)])  # the same sentence in two batches
    batch_b = torch.stack([sentence, torch.randn(T, d) * 5 + 2])
    bn = torch.nn.BatchNorm1d(d, affine=False)  # statistics over the batch, per feature

    def ln(batch):  # the first sentence's output under layer norm
        return layer_norm(batch, 1.0, 0.0)[0]

    def batch_norm(batch):  # ...and under batch norm, over every token in the batch
        return bn(batch.reshape(-1, d)).reshape(batch.shape)[0]

    assert torch.allclose(ln(batch_a), ln(batch_b))  # layer norm: only the sentence matters
    assert not torch.allclose(batch_norm(batch_a), batch_norm(batch_b))  # batch norm: neighbours leak in
    print("layer norm ignores the rest of the batch; batch norm doesn't")
