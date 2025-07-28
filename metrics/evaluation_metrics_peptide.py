import torch
from metrics.chamfer_distance import chamfer_distance
from metrics.earth_mover_distance import earth_mover_distance
from metrics.utils.lgan_mmd_cov import lgan_mmd_cov
from metrics.utils.knn import knn


def pairwise_cd_matrix(pcs1, pcs2, batch_size):
    """
    Return pairwise CD matrix: (len(pcs1), len(pcs2))
    """
    pcs1_list = pcs1.split(1, dim=0)  # B1 x (1, 3, N)
    pcs2_list = pcs2.split(1, dim=0)  # B2 x (1, 3, N)

    matrix = torch.zeros(len(pcs1_list), len(pcs2_list))
    for i, p1 in enumerate(pcs1_list):
        for j, p2 in enumerate(pcs2_list):
            cd = chamfer_distance(p1.squeeze(0).unsqueeze(0), p2.squeeze(0).unsqueeze(0))  # (1, 3, N)
            matrix[i, j] = cd
    return matrix

def pairwise_emd_matrix(pcs1, pcs2, batch_size):
    pcs1_list = pcs1.split(1, dim=0)
    pcs2_list = pcs2.split(1, dim=0)

    matrix = torch.zeros(len(pcs1_list), len(pcs2_list))
    for i, p1 in enumerate(pcs1_list):
        for j, p2 in enumerate(pcs2_list):
            emd = earth_mover_distance(p1.squeeze(0).unsqueeze(0), p2.squeeze(0).unsqueeze(0))  # (1, 3, N)
            matrix[i, j] = emd
    return matrix


def compute_all_metrics_peptide(sample_pcs, ref_pcs, batch_size):
    """
    sample_pcs: Tensor of shape (B, 3, N) — generated point clouds
    ref_pcs:    Tensor of shape (B, 3, N) — reference point clouds
    batch_size: mini-batch size for pairwise EMD/CD

    Returns: dict of metrics: MMD, COV, 1NN for CD & EMD
    """
    results = {}

    # === Step 1: Pairwise distance matrix ===
    M_rs_cd = pairwise_cd_matrix(ref_pcs, sample_pcs, batch_size)
    M_rs_emd = pairwise_emd_matrix(ref_pcs, sample_pcs, batch_size)

    # === Step 2: MMD / COV ===
    results.update({f"{k}-CD": v for k, v in lgan_mmd_cov(M_rs_cd.t()).items()})
    results.update({f"{k}-EMD": v for k, v in lgan_mmd_cov(M_rs_emd.t()).items()})

    # === Step 3: 1-NN ===
    M_rr_cd = pairwise_cd_matrix(ref_pcs, ref_pcs, batch_size)
    M_ss_cd = pairwise_cd_matrix(sample_pcs, sample_pcs, batch_size)
    one_nn_cd = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({f"1-NN-CD-{k}": v for k, v in one_nn_cd.items() if "acc" in k})

    M_rr_emd = pairwise_emd_matrix(ref_pcs, ref_pcs, batch_size)
    M_ss_emd = pairwise_emd_matrix(sample_pcs, sample_pcs, batch_size)
    one_nn_emd = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    results.update({f"1-NN-EMD-{k}": v for k, v in one_nn_emd.items() if "acc" in k})

    return results
