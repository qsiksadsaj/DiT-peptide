import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from datasets.peptide_data import PeptidePointClouds
from models.dit3d_notrans import DiT3D_models
from train_pp import peptide_collate_fn
from torch.nn.utils.rnn import pad_sequence

def normalize_to_unit_box(points):
    center = np.mean(points, axis=0)
    shifted = points - center
    max_range = np.abs(shifted).max()
    return shifted / max_range


@torch.no_grad()
def evaluate_voxel_consistency(model, dataloader, device, output_dir="./output_xyz", save_samples_path="./checkpoints/samples_notrans.pth"):
    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    sample_index = 0
    samples_list = []

    for batch in tqdm(dataloader, desc="Saving GT & Pred xyz"):
        x = batch["test_points"].to(device).transpose(1, 2)  # [B, 3, N]
        mask = batch["test_masks"].to(device)                # [B, N]
        y = batch["cate_idx"].to(device)                      # [B]
        t = torch.zeros_like(y)                               # [B], dummy timestep

        # Skip transformer, just voxel→point流程
        x_recon = model(x, t, y, mask=mask, skip_network=True)  # [B, 3, N]

        for i in range(x.shape[0]):
            gt = x[i].transpose(0, 1).cpu().numpy()
            gt = normalize_to_unit_box(gt)
            pred = x_recon[i].transpose(0, 1).cpu().numpy()

            mid_value = batch['mid'][i]
            if isinstance(mid_value, (list, tuple)) and len(mid_value) == 1:
                sample_id = mid_value[0]
            elif isinstance(mid_value, bytes):
                sample_id = mid_value.decode("utf-8")
            else:
                sample_id = str(mid_value)

            sample_id = mid_value.split("/")[-1]
            # print(f"Processing sample {sample_id}...")
            sample_folder = os.path.join(output_dir, sample_id)
            os.makedirs(sample_folder, exist_ok=True)

            np.savetxt(os.path.join(sample_folder, "gt.xyz"), gt, fmt="%.6f")
            np.savetxt(os.path.join(sample_folder, "pred.xyz"), pred, fmt="%.6f")

            # === 保存样本信息到 samples_list ===
            samples_list.append({
                "mid": sample_id,
                "gen_coords": torch.from_numpy(pred).float()  # shape [N, 3]
            })

    # === 保存为 .pth 文件 ===
    torch.save(samples_list, save_samples_path)
    print(f"✅ Saved {len(samples_list)} samples to {save_samples_path}")


def get_test_loader(root, npoints, batch_size=1):
    dataset = PeptidePointClouds(
        root_dir=root,
        split='test',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.0,
        normalize_per_shape=True,
        normalize_std_per_axis=False,
        random_subsample=False
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=peptide_collate_fn)
    return dataloader


def compute_rmsd_from_samples(samples_pth_path):
    import numpy as np
    import torch
    import os
    from Bio.PDB import PDBParser

    def pdb_to_ca_point_cloud(pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pep", pdb_path)
        coords = [atom.get_coord() for atom in structure.get_atoms() if atom.get_id() == "CA"]
        return np.array(coords)

    def compute_rmsd(pred, gt):
        diff = pred - gt
        return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    def kabsch_align(pred, gt):
        pred_center = pred.mean(0)
        gt_center = gt.mean(0)
        pred_centered = pred - pred_center
        gt_centered = gt - gt_center
        H = pred_centered.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        return (pred_centered @ R) + gt_center

    meta_dict_path = "/root/DiT-3D/data/peptides_only_ligand_raw/peptide/meta_dict_test.npy"
    # meta_dict_path = "/root/DiT-3D/data/peptides_only_ligand_raw_no_norm/peptide/test/meta_dict.npy"
    raw_pdb_root = "/root/DiT-3D/data/peptide_dataset_xj/PPDbench_xj_10_repaired"
    meta_dict = np.load(meta_dict_path, allow_pickle=True).item()
    samples_list = torch.load(samples_pth_path)

    rmsd_list = []

    for i, item in enumerate(samples_list):
        mid = item["mid"]
        gen_coords = item["gen_coords"].numpy()
        sample_id = mid.split("/")[-1]

        if sample_id not in meta_dict:
            print(f"❌ sample_id {sample_id} 不在 meta_dict 中！跳过")
            continue

        center = np.array(meta_dict[sample_id]["center"])
        scale = meta_dict[sample_id]["scale"]
        # center = np.array([0.0, 0.0, 0.0])
        # scale = 1.0
        print(center)
        gen_coords_denorm = gen_coords * scale + center

        pdb_path = os.path.join(raw_pdb_root, sample_id, "peptide_repaired.pdb")
        if not os.path.exists(pdb_path):
            print(f"❌ PDB 文件不存在: {pdb_path}")
            continue

        ca_coords_gt = pdb_to_ca_point_cloud(pdb_path)
        if ca_coords_gt.shape != gen_coords_denorm.shape:
            print(f"⚠️ CA 数量不一致: GT={ca_coords_gt.shape} vs GEN={gen_coords_denorm.shape}")
            continue

        rmsd_raw = compute_rmsd(gen_coords_denorm, ca_coords_gt)
        aligned = kabsch_align(gen_coords_denorm, ca_coords_gt)
        rmsd_aligned = compute_rmsd(aligned, ca_coords_gt)

        print(f"[{i:03d}] {mid} RMSD: raw = {rmsd_raw:.3f} Å → aligned = {rmsd_aligned:.3f} Å")
        rmsd_list.append(rmsd_aligned)

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

def main():
    # === 模拟 args（你也可以用 argparse 来写） ===
    class Args:
        voxel_size = 16
        model_type = "DiT-S/4"
        num_classes = 55
        use_pretrained = False

    opt = Args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 模型初始化 ===
    model = DiT3D_models[opt.model_type](
        input_size=opt.voxel_size,
        num_classes=opt.num_classes,
        pretrained=opt.use_pretrained
    )

    # === 数据加载 ===
    dataloader = get_test_loader(
        root="/root/DiT-3D/data/peptides_only_ligand_raw",
        npoints=128,
        batch_size=1
    )

    save_path = "./checkpoints/samples_notrans.pth"
    
    # === 评估转换精度 ===
    evaluate_voxel_consistency(
    model=model,
    dataloader=dataloader,
    device=device,
    output_dir="./debug_test_reconstruction_notrans",
    save_samples_path=save_path
    )

    compute_rmsd_from_samples(save_path)




if __name__ == "__main__":
    main()




# import os
# import torch
# from tqdm import tqdm
# import numpy as np

# from train_pp import get_dataset, get_dataloader, peptide_collate_fn
# from models.dit3d import DiT3D_models


# @torch.no_grad()
# def evaluate_voxel_consistency(model, dataloader, device, output_dir="./debug_test_reconstruction_notrans"):
#     model.eval()
#     model.to(device)

#     os.makedirs(output_dir, exist_ok=True)
#     sample_index = 0

#     for batch in tqdm(dataloader, desc="Saving GT & Pred xyz"):
#         x = batch["train_points"].to(device).transpose(1, 2)  # [B, 3, N]
#         mask = batch["train_masks"].to(device)                # [B, N]
#         y = batch["cate_idx"].to(device)                      # [B]
#         t = torch.zeros_like(y)                               # [B], dummy timestep

#         # === skip transformer blocks, only voxelization chain ===
#         x_recon = model(x, t, y, mask=mask, skip_network=True)  # [B, 3, N]

#         for i in range(x.shape[0]):
#             gt = x[i].transpose(0, 1).cpu().numpy()
#             pred = x_recon[i].transpose(0, 1).cpu().numpy()

#             mid_value = batch['mid'][i]
#             if isinstance(mid_value, (list, tuple)) and len(mid_value) == 1:
#                 sample_id = mid_value[0]
#             elif isinstance(mid_value, bytes):
#                 sample_id = mid_value.decode("utf-8")
#             else:
#                 sample_id = str(mid_value)

#             sample_id = sample_id.replace("/", "_")
#             # print(f"Processing sample {sample_id}...")
#             sample_folder = os.path.join(output_dir, sample_id)
#             os.makedirs(sample_folder, exist_ok=True)

#             np.savetxt(os.path.join(sample_folder, "gt.xyz"), gt, fmt="%.6f")
#             np.savetxt(os.path.join(sample_folder, "pred.xyz"), pred, fmt="%.6f")


# def main():
#     # === Minimal Args Placeholder (mimic train_pp structure) ===
#     class Args:
#         dataroot = "/root/DiT-3D/data/peptides_only_ligand_raw"
#         category = "peptide"
#         npoints = 128
#         voxel_size = 16
#         model_type = "DiT-S/4"
#         num_classes = 55
#         use_pretrained = False
#         bs = 1
#         workers = 4
#         distribution_type = None
#         world_size = 1
#         rank = 0

#     opt = Args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # === Get dataset & dataloader ===
#     train_dataset, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)
#     _, test_loader, _, _ = get_dataloader(opt, train_dataset=test_dataset, test_dataset=test_dataset)

#     # === Init model ===
#     model = DiT3D_models[opt.model_type](
#         input_size=opt.voxel_size,
#         num_classes=opt.num_classes,
#         pretrained=opt.use_pretrained
#     )

#     # === Evaluate: Save xyz files ===
#     evaluate_voxel_consistency(
#         model=model,
#         dataloader=test_loader,
#         device=device,
#         output_dir="./debug_test_reconstruction_notrans"
#     )


# if __name__ == "__main__":
#     main()
