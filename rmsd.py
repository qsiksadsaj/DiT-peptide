import torch
import numpy as np
from Bio.PDB import PDBParser
import os

def pdb_to_ca_point_cloud(pdb_path):
    """
    从 PDB 提取所有 CA 坐标，返回 ndarray (N, 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", pdb_path)
    coords = []
    for atom in structure.get_atoms():
        if atom.get_id() == "CA":
            coords.append(atom.get_coord())
    return np.array(coords)

def write_ca_coords_to_pdb(ca_coords, out_path, chain_id="A", start_residue=1, residue_name="GLY"):
    """
    将 CA 坐标写出为 PDB 文件
    """
    with open(out_path, "w") as f:
        for i, coord in enumerate(ca_coords, start=start_residue):
            x, y, z = coord
            record = (
                f"ATOM  {i:5d}  CA  {residue_name:<3s} {chain_id}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  1.00           C\n"
            )
            f.write(record)
        f.write("END\n")
    print(f"✅ Wrote PDB to {out_path}")

def compute_rmsd(pred, gt):
    """
    计算 RMSD
    """
    if len(pred) != len(gt):
        raise ValueError(f"长度不一致: pred={len(pred)} vs gt={len(gt)}")
    diff = pred - gt
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd

def kabsch_align(pred, gt):
    """
    使用 Kabsch 算法将 pred 对齐到 gt（输入形状均为 [N, 3]）

    返回对齐后的 pred
    """
    assert pred.shape == gt.shape, "Kabsch 对齐要求 pred 和 gt 的形状一致"

    # 去中心化
    pred_center = pred.mean(axis=0)
    gt_center = gt.mean(axis=0)
    pred_centered = pred - pred_center
    gt_centered = gt - gt_center

    # 协方差矩阵
    H = pred_centered.T @ gt_centered

    # SVD 分解
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 防止反射（保留右手坐标系）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 对 pred 应用旋转 + 平移
    aligned_pred = (pred_centered @ R) + gt_center

    return aligned_pred


if __name__ == "__main__":
    # 路径配置
    samples_pth_path = "/root/DiT-3D/checkpoints/dit3d_raw/syn/samples_19999.pth"  # 这个存的结果是 forward_dense_no_nosie 的
    # samples_pth_path = "/root/DiT-3D/checkpoints/samples_notrans.pth"
    meta_dict_path = "/root/DiT-3D/data/peptides_only_ligand_raw_no_norm/peptide/scale_center_test.npy"
    raw_pdb_root = "/root/DiT-3D/data/peptides_dataset_xj/PPDbench_xj_10_repaired"

    # 加载 meta_dict
    meta_dict = np.load(meta_dict_path, allow_pickle=True).item()

    rmsd_list = []

    # 加载 samples.pth
    samples_list = torch.load(samples_pth_path)
    print(f"✅ Loaded samples: {len(samples_list)}")

    for i, item in enumerate(samples_list):

        mid = item["mid"]     # e.g. test/4k0u
        gen_coords = item["gen_coords"].numpy()    # (N, 3)

        # 提取 sample_id
        sample_id = mid.split("/")[-1]
        print(f"🔍 解析出的 sample_id: {sample_id}")
        if sample_id not in meta_dict:
            print(f"❌ sample_id {sample_id} 不在 meta_dict 中！跳过")
            continue

        center = np.array(meta_dict[sample_id]["center"])
        scale = meta_dict[sample_id]["scale"]

        # 反归一化
        gen_coords_denorm = gen_coords * scale + center

        # 读取原始 PDB 文件
        pdb_path = os.path.join(raw_pdb_root, sample_id, "peptide_repaired.pdb")

        if not os.path.exists(pdb_path):
            print(f"❌ PDB 文件不存在: {pdb_path}")
            continue

        ca_coords_gt = pdb_to_ca_point_cloud(pdb_path)

        if ca_coords_gt.shape != gen_coords_denorm.shape:
            print(f"⚠️ CA 数量不一致: GT={ca_coords_gt.shape} vs GEN={gen_coords_denorm.shape}")
            continue

        # 计算 RMSD
        np.set_printoptions(precision=2, suppress=True)

        # print(gen_coords_denorm)
        # print(ca_coords_gt)
        # rmsd = compute_rmsd(gen_coords_denorm, ca_coords_gt)
        rmsd_raw = compute_rmsd(gen_coords_denorm, ca_coords_gt)
        gen_coords_aligned = kabsch_align(gen_coords_denorm, ca_coords_gt)
        rmsd_aligned = compute_rmsd(gen_coords_aligned, ca_coords_gt)

        print(f"[{i:03d}] {mid} RMSD: raw = {rmsd_raw:.3f} Å → aligned = {rmsd_aligned:.3f} Å")

        rmsd_list.append(rmsd_aligned)

    # ===== 汇总统计 =====

    if len(rmsd_list) > 0:
        rmsd_array = np.array(rmsd_list)

        print("\n=== RMSD Summary ===")
        print(f"样本数量: {len(rmsd_array)}")
        print(f"均值 (Mean RMSD): {rmsd_array.mean():.3f} Å")
        print(f"标准差 (Std): {rmsd_array.std():.3f} Å")
        print(f"最小值 (Min): {rmsd_array.min():.3f} Å")
        print(f"最大值 (Max): {rmsd_array.max():.3f} Å")
        print(f"中位数 (Median): {np.median(rmsd_array):.3f} Å")
        print(f"25% 分位数: {np.percentile(rmsd_array, 25):.3f} Å")
        print(f"75% 分位数: {np.percentile(rmsd_array, 75):.3f} Å")

    else:
        print("⚠️ 没有计算出 RMSD。")