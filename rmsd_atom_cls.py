import os
import torch
import numpy as np
from Bio.PDB import PDBParser
from sklearn.cluster import DBSCAN

def cluster_atoms_from_point_cloud(points, center, scale, atom_thresholds=None):
    points = points * scale + center  # un-normalize
    if atom_thresholds is None:
        atom_thresholds = {
            "CA": (50, 70),
            "N": (30, 50),
            "O": (10, 30),
        }
    clustering = DBSCAN(eps=0.8, min_samples=5).fit(points)
    labels = clustering.labels_
    coords, types = [], []
    for k in set(labels):
        if k == -1:
            continue
        cluster = points[labels == k]
        count = len(cluster)
        for atom_type, (low, high) in atom_thresholds.items():
            if low <= count <= high:
                coords.append(cluster.mean(axis=0))
                types.append(atom_type)
                break
    return np.array(coords), types

def parse_pdb_atoms(pdb_path, allowed=["CA", "N", "O"]):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("peptide", pdb_path)
    coords, types = [], []
    for atom in structure.get_atoms():
        if atom.get_id() in allowed:
            coords.append(atom.coord)
            types.append(atom.get_id())
    return np.array(coords), types

def kabsch_align(P, Q):
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    H = P_cent.T @ Q_cent
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return (P_cent @ R) + Q.mean(axis=0)

def compute_rmsd(pred, gt):
    return np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1)))

# 路径配置
samples_pth = "/root/DiT-3D/checkpoints/dit3d_atom_cls/syn/samples.pth"
meta_dict = np.load("/root/DiT-3D/data/peptides_only_ligand_atom_cls/peptide/meta_dict_test.npy", allow_pickle=True).item()
raw_pdb_root = "/root/DiT-3D/data/peptide_dataset_xj/PPDbench_xj_10_repaired"

samples = torch.load(samples_pth)
print(f"✅ Loaded {len(samples)} samples.")

rmsd_all = []

for i, sample in enumerate(samples):
    mid = sample["mid"]
    sample_id = mid.split("/")[-1]
    if sample_id not in meta_dict:
        print(f"⚠️ {sample_id} not in meta_dict, skipped.")
        continue

    pdb_path = os.path.join(raw_pdb_root, sample_id, "peptide_repaired.pdb")
    if not os.path.exists(pdb_path):
        print(f"❌ PDB not found: {pdb_path}")
        continue

    center = np.array(meta_dict[sample_id]["center"])
    scale = meta_dict[sample_id]["scale"]
    pred_points = sample["gen_coords"].numpy()  # dense point cloud

    pred_coords, pred_types = cluster_atoms_from_point_cloud(pred_points, center, scale)
    gt_coords, gt_types = parse_pdb_atoms(pdb_path)

    if sorted(pred_types) != sorted(gt_types):
        print(f"⚠️ Atom type mismatch: pred={pred_types}, gt={gt_types}")
        continue

    # 匹配顺序（按 type 逐个对应）
    pred_aligned, gt_sorted = [], []
    for at in ["CA", "N", "O"]:
        for pc, pt in zip(pred_coords, pred_types):
            if pt == at:
                pred_aligned.append(pc)
        for gc, gt in zip(gt_coords, gt_types):
            if gt == at:
                gt_sorted.append(gc)

    pred_aligned = np.array(pred_aligned)
    gt_sorted = np.array(gt_sorted)

    if pred_aligned.shape != gt_sorted.shape:
        print(f"⚠️ Skipped {sample_id}: shape mismatch")
        continue

    pred_kabsch = kabsch_align(pred_aligned, gt_sorted)
    rmsd = compute_rmsd(pred_kabsch, gt_sorted)
    print(f"[{i:03d}] {sample_id}: RMSD = {rmsd:.3f} Å")
    rmsd_all.append(rmsd)

# 汇总
if rmsd_all:
    rmsd_all = np.array(rmsd_all)
    print("\n📊 RMSD Summary")
    print(f"Count: {len(rmsd_all)}")
    print(f"Mean: {rmsd_all.mean():.3f} Å")
    print(f"Std: {rmsd_all.std():.3f} Å")
    print(f"Min: {rmsd_all.min():.3f} Å")
    print(f"Max: {rmsd_all.max():.3f} Å")
    print(f"Median: {np.median(rmsd_all):.3f} Å")
else:
    print("⚠️ No RMSD computed.")
