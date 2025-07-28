import torch.nn as nn
import torch
import modules.functional as F

__all__ = ['KLLoss']


class KLLoss(nn.Module):
    def forward(self, x, y):
        return F.kl_loss(x, y)


def label_distribution_loss(pred_logits, labels, mask=None, eps=1e-8):
    """
    pred_logits: [B, C, N]
    labels: [B, N]         (int)
    mask: [B, N] or None
    """
    B, C, N = pred_logits.shape

    # softmax → [B, C, N]
    probs = torch.nn.functional.softmax(pred_logits, dim=1)

    if mask is not None:
        mask = mask.float()                     # [B, N]
        mask_sum = mask.sum(dim=1, keepdim=True) + eps
        probs = probs * mask.unsqueeze(1)       # mask out padded points
    else:
        mask_sum = torch.tensor(N, device=labels.device).float().expand(B, 1)

    # === pred label distribution ===
    hist_pred = probs.sum(dim=2) / mask_sum    # [B, C]

    # === GT label distribution ===
    hist_gt = []
    for b in range(B):
        if mask is not None:
            lbl = labels[b][mask[b] > 0]
        else:
            lbl = labels[b]
        hist = torch.bincount(lbl, minlength=C).float()
        hist /= (hist.sum() + eps)
        hist_gt.append(hist)
    hist_gt = torch.stack(hist_gt, dim=0)      # [B, C]

    # === MSE loss ===
    loss = torch.nn.functional.mse_loss(hist_pred, hist_gt)

    return loss

def spatial_affinity_loss(pred_logits, pred_points, mask=None, sigma=1.0, eps=1e-8):
    """
    pred_logits: [B, C, N]
    pred_points: [B, 3, N]
    mask: [B, N] or None
    """
    B, C, N = pred_logits.shape

    # softmax → [B, C, N]
    probs = torch.nn.functional.softmax(pred_logits, dim=1)  # [B, C, N]

    # pairwise KL divergence between all points
    probs_i = probs.unsqueeze(3)     # [B, C, N, 1]
    probs_j = probs.unsqueeze(2)     # [B, C, 1, N]

    kl_ij = (probs_i * (probs_i / (probs_j + eps)).log()).sum(dim=1)   # [B, N, N]

    # compute pairwise distances
    points = pred_points.transpose(1, 2)       # [B, N, 3]
    dists = torch.cdist(points, points, p=2) ** 2     # [B, N, N]

    # affinity weights
    affinity = torch.exp(-dists / (2 * sigma**2))    # [B, N, N]

    # mask out padding regions
    if mask is not None:
        mask = mask.float()                    # [B, N]
        mask_i = mask.unsqueeze(2)             # [B, N, 1]
        mask_j = mask.unsqueeze(1)             # [B, 1, N]
        affinity = affinity * (mask_i * mask_j)
        kl_ij = kl_ij * (mask_i * mask_j)

    weighted_kl = affinity * kl_ij
    loss = weighted_kl.sum() / (affinity.sum() + eps)

    return loss

