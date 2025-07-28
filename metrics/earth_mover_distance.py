import torch

def earth_mover_distance(p1, p2, eps=0.01, max_iter=100):
    """
    p1, p2: (1, 3, N)
    Return scalar EMD (Sinkhorn approximation)
    """
    x, y = p1.squeeze(0).transpose(0, 1), p2.squeeze(0).transpose(0, 1)  # (N, 3)
    cost = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0)  # (N, N)
    
    N = cost.size(0)
    mu = torch.full((N,), 1.0 / N, device=cost.device)
    nu = torch.full((N,), 1.0 / N, device=cost.device)

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    K = torch.exp(-cost / eps)

    for _ in range(max_iter):
        u = mu / (K @ v + 1e-8)
        v = nu / (K.T @ u + 1e-8)

    T = torch.outer(u, v) * K
    emd = torch.sum(T * cost)
    return emd
