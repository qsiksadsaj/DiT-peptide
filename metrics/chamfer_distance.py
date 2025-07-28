import torch

def chamfer_distance(p1, p2):
    """
    p1: (1, 3, N)
    p2: (1, 3, N)
    Return scalar CD
    """
    x, y = p1.squeeze(0).transpose(0, 1), p2.squeeze(0).transpose(0, 1)  # (N, 3)
    dist1 = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0)  # (N, N)
    cd = dist1.min(dim=1)[0].mean() + dist1.min(dim=0)[0].mean()
    return cd
