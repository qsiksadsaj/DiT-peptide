import torch

def knn(M_rr, M_rs, M_ss, k=1, sqrt=False):
    """
    M_rr: (B, B) — gt-gt 距离
    M_rs: (B, B) — gt-pred 距离
    M_ss: (B, B) — pred-pred 距离
    Return: dict{"acc": float}
    """
    B = M_rr.shape[0]
    labels = torch.cat([torch.zeros(B), torch.ones(B)])  # 0=gt, 1=gen
    feats = torch.cat([
        torch.cat([M_rr, M_rs], dim=1),
        torch.cat([M_rs.t(), M_ss], dim=1)
    ], dim=0)

    feats.fill_diagonal_(float("inf"))  # 忽略自身

    _, nn_idx = feats.topk(k, largest=False, dim=1)
    nn_labels = labels[nn_idx]  # (2B, k)
    pred = (nn_labels.mean(dim=1) > 0.5).float()
    acc = (pred == labels).float().mean()

    return {
        "acc": acc
    }
