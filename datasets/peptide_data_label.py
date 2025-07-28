import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import open3d as o3d

cate_to_synsetid = {"peptide": "peptide"}
synsetid_to_cate = {"peptide": "peptide"}
class Peptide(Dataset):
    def __init__(self,
                 root_dir,
                 synset_ids,
                 tr_sample_size=2048,
                 te_sample_size=2048,
                 split='train',
                 scale=1.,
                 normalize_per_shape=True,
                 box_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None,
                 all_points_std=None,
                 input_dim=3,
                 use_mask=False):
        self.root_dir = root_dir
        self.subdirs = synset_ids
        self.split = split
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.scale = scale
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.use_mask = use_mask
        self.box_per_shape = box_per_shape
        

        self.all_points = []
        self.all_labels = []
        self.all_center = []
        self.all_scale = []
        self.all_ca_coords = []
        self.cate_idx_lst = []
        self.all_cate_mids = []

        for cate_idx, subd in enumerate(self.subdirs):
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print(f"Directory missing : {sub_path}")
                continue

            for fname in os.listdir(sub_path):
                if not fname.endswith('.npy'):
                    continue
                mid = os.path.join(self.split, fname[:-4])
                npy_path = os.path.join(root_dir, subd, mid + '.npy')

                try:
                    data = np.load(npy_path, allow_pickle=True).item()

                    points = data["points"]            # (N, 3)
                    labels = data["labels"]            # (N,)
                    center = data["center"]            # (3,)
                    scale_val = data["scale"]          # float
                    ca_coords = data["ca_coords"]      # (M, 3)

                    # shape check
                    assert points.shape[0] == labels.shape[0]

                    self.all_points.append(points[np.newaxis, ...])
                    self.all_labels.append(labels[np.newaxis, ...])
                    self.all_center.append(center)
                    self.all_scale.append(scale_val)
                    self.all_ca_coords.append(ca_coords)

                    self.cate_idx_lst.append(cate_idx)
                    self.all_cate_mids.append((subd, mid))

                except Exception as e:
                    print(f"❌ Error loading {npy_path}: {e}")
                    continue

        if len(self.all_points) == 0:
            raise ValueError(f"No npy files found in {root_dir}")

        # Shuffle
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)

        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_labels = [self.all_labels[i] for i in self.shuffle_idx]
        self.all_center = [self.all_center[i] for i in self.shuffle_idx]
        self.all_scale = [self.all_scale[i] for i in self.shuffle_idx]
        self.all_ca_coords = [self.all_ca_coords[i] for i in self.shuffle_idx]
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        if self.scale != 1.:
            for i in range(len(self.all_points)):
                self.all_points[i][0] *= self.scale

        print(f"✅ Loaded {len(self.all_points)} peptides from {root_dir}/{self.split}")

    def __len__(self):
        return len(self.all_points)

    def __getitem__(self, idx):
        points = self.all_points[idx][0]             # (N, 3)
        labels = self.all_labels[idx][0]             # (N,)
        center = self.all_center[idx]                # (3,)
        scale = self.all_scale[idx]                  # float
        ca_coords = self.all_ca_coords[idx]          # (M, 3)

        # optional normalize again
        if self.normalize_per_shape:
            points = (points - center) / (scale + 1e-8)

        N = points.shape[0]
        target_N = self.tr_sample_size

        if N >= target_N:
            sampled_idxs = torch.randperm(N)[:target_N]
        else:
            reps = target_N // N
            rem = target_N % N
            sampled_idxs = torch.cat([
                torch.arange(N).repeat(reps),
                torch.randint(0, N, (rem,))
            ])

        points_sampled = points[sampled_idxs, :]
        labels_sampled = labels[sampled_idxs]

        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        out = {
            'idx': idx,
            'train_points': torch.from_numpy(points_sampled).float(),
            'train_labels': torch.from_numpy(labels_sampled).long(),
            'test_points': torch.from_numpy(points_sampled).float(),
            'test_labels': torch.from_numpy(labels_sampled).long(),
            'center': torch.from_numpy(center).float(),
            'scale': torch.tensor(scale).float(),
            'ca_coords': torch.from_numpy(ca_coords).float(),
            'cate_idx': cate_idx,
            'sid': sid,
            'mid': mid
        }

        return out



class PeptidePointClouds(Peptide):
    def __init__(self,
                 root_dir="data/peptides",
                 categories=['peptide'],
                 tr_sample_size=2048,
                 te_sample_size=2048,
                 split='train',
                 scale=1.,
                 normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 box_per_shape=False,
                 random_subsample=False,
                 all_points_mean=None,
                 all_points_std=None,
                 use_mask=False):
        """
        categories: list like ['peptide'] 
        """

        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size

        # peptide task 暂时只有一个类别
        self.cates = categories

        # 因为 peptide 数据都在 peptide 目录下 → synset_ids 就是 ['peptide']
        if 'all' in categories:
            self.synset_ids = ['peptide']
        else:
            self.synset_ids = categories

        # gravity_axis, display_axis_order → 你原本有
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        # 调用父类初始化
        super(PeptidePointClouds, self).__init__(
            root_dir=root_dir,
            synset_ids=self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            box_per_shape=box_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
            input_dim=3,
            use_mask=use_mask
        )

from torch.nn.utils.rnn import pad_sequence

def peptide_collate_fn(batch):
    out = {}

    # === train_points
    train_points_list = [b["train_points"] for b in batch]
    train_points_padded = pad_sequence(
        train_points_list, batch_first=True, padding_value=0.0
    )
    out["train_points"] = train_points_padded

    # === test_points
    test_points_list = [b["test_points"] for b in batch]
    test_points_padded = pad_sequence(
        test_points_list, batch_first=True, padding_value=0.0
    )
    out["test_points"] = test_points_padded

    # === train_labels
    train_labels_list = [b["train_labels"] for b in batch]
    train_labels_padded = pad_sequence(
        train_labels_list, batch_first=True, padding_value=0
    )
    out["train_labels"] = train_labels_padded

    # === test_labels
    test_labels_list = [b["test_labels"] for b in batch]
    test_labels_padded = pad_sequence(
        test_labels_list, batch_first=True, padding_value=0
    )
    out["test_labels"] = test_labels_padded

    # === lengths
    lengths = torch.tensor([x.shape[0] for x in train_points_list], dtype=torch.long)
    out["lengths"] = lengths

    # === masks
    max_len = train_points_padded.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    out["train_masks"] = mask
    out["test_masks"] = mask

    # === other keys
    for key in batch[0].keys():
        if key in [
            "train_points",
            "test_points",
            "train_labels",
            "test_labels",
        ]:
            continue

        values = [b[key] for b in batch]

        if isinstance(values[0], np.ndarray):
            out[key] = torch.stack([torch.from_numpy(v).float() for v in values])
        elif isinstance(values[0], torch.Tensor):
            try:
                out[key] = torch.stack(values)
            except:
                out[key] = values
        elif isinstance(values[0], (int, float)):
            out[key] = torch.tensor(values)
        else:
            out[key] = values

    return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 创建 dataset
    dataset = PeptidePointClouds(
        root_dir="/root/DiT-3D/data/peptides_only_ligand_label",
        categories=["peptide"],
        split="train",
        tr_sample_size=2048,
        te_sample_size=2048,
        random_subsample=True
    )

    print(f"Dataset size: {len(dataset)}")

    # 查看第一条数据
    sample = dataset[0]

    print("\n=== Sample keys and shapes ===")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        elif isinstance(v, (list, tuple)):
            print(f"{k}: list/tuple of len {len(v)}")
        else:
            print(f"{k}: {v}")

    # 用 DataLoader 测试
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=peptide_collate_fn,    # <<<< 关键修改
    )

    batch = next(iter(loader))
    print("\n=== Batch keys and shapes ===")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        elif isinstance(v, (list, tuple)):
            print(f"{k}: list/tuple of len {len(v)}")
        else:
            print(f"{k}: {v}")
