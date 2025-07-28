def lgan_mmd_cov(dist_mat):
    """
    dist_mat: (B_gt, B_gen)
    Return:
        - MMD: 最小距离均值
        - COV: 被覆盖的真实样本比例
    """
    min_distances = dist_mat.min(dim=0)[0]  # for MMD
    mmd = min_distances.mean()

    cov = (dist_mat.min(dim=1)[1].unique().numel() / dist_mat.size(0))

    return {
        "MMD": mmd,
        "COV": cov
    }
