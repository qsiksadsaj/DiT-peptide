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
                 normalize_per_shape=True,  # peptide不定长 默认 样本自身归一化
                 box_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None,
                 all_points_std=None,
                 input_dim=3,
                 use_mask=False):
        """
        synset_ids: list of categories e.g. ['peptide']
        """

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
        if use_mask:
            self.mask_transform = PointCloudMasks(radius=5, elev=5, azim=90)

        # ===== Load all npy =====
        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []

        for cate_idx, subd in enumerate(self.subdirs):  # 遍历类别
            sub_path = os.path.join(root_dir, subd, self.split)  # 拼接类别路径
            if not os.path.isdir(sub_path):
                print(f"Directory missing : {sub_path}")
                continue

            all_mids = []  # 找到该类别下所有.npy 文件的模型 id   一个类别的 train/test/val 下面有许多.npy 文件，文件名就是 id   
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                mid = os.path.join(self.split, x[:-4])  # 去掉 .npy
                all_mids.append(mid)

            for mid in all_mids:  # 逐个加载 npy 文件
                npy_path = os.path.join(root_dir, subd, mid + '.npy')
                try:
                    points = np.load(npy_path)  # (N, 3)
                except:
                    continue

                self.all_points.append(points[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        if len(self.all_points) == 0:
            raise ValueError(f"No npy files found in {root_dir}")

        # Shuffle
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Concatenate 因为 ShapeNet 都是 15000 可以这么做  peptide 不是 不做
        # self.all_points = np.concatenate(self.all_points)  # (B, N, 3)

        # Normalization 这里和 ShapeNet 不同 peptide 数据集每个类别的点云数量不一样 不能变成 np 直接 reshape 单独每个样本做 normalization
        if all_points_mean is not None and all_points_std is not None:
            # Use pre-computed mean and std
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif normalize_per_shape:  # 每个样本单独归一化
            self.all_points_mean = []
            self.all_points_std = []

            for pts in self.all_points:
                pts_ = pts[0]      # (N, 3)
                mean = pts_.mean(axis=0, keepdims=True)    # (1, 3)

                if normalize_std_per_axis:
                    std = pts_.std(axis=0, keepdims=True)  # (1, 3)
                else:
                    std = np.array([[pts_.std()]], dtype=np.float32)  # (1, 1)

                self.all_points_mean.append(mean)
                self.all_points_std.append(std)

            self.all_points_mean = np.stack(self.all_points_mean, axis=0)   # (B, 1, 3) 各自归一化 各自的 mean/std
            self.all_points_std = np.stack(self.all_points_std, axis=0)  # (B, 1, 3)  各自归一化 各自的 mean/std
        else:  # 全局归一化  但是不定长 没想好怎么写 这里先空着
            pass

        # apply normalization 同理这里也改
        for i in range(len(self.all_points)):
            pts = self.all_points[i][0]    # (N, 3)
            m = self.all_points_mean[i]
            s = self.all_points_std[i]
            self.all_points[i][0] = (pts - m) / (s + 1e-8)    # avoid divide by zero
        # if self.box_per_shape: 
        # 忽略point_level的split   ShapeNet里都是15000 保证前10000里加噪 后5000在test_point里  但peptide不定长  忽略了train_points 和 test_points 的切分
        # 改成在 __getitem__ 中随机切分  
        #     self.all_points = self.all_points - 0.5
        # self.train_points = self.all_points[:, :10000]  # 切分训练集 测试集
        # self.test_points = self.all_points[:, 10000:]

        # self.tr_sample_size = min(10000, tr_sample_size)  # 限制采样大小
        # self.te_sample_size = min(5000, te_sample_size)
        # print("Total number of data:%d" % len(self.train_points))
        # print("Min number of points: (train)%d (test)%d"
        #       % (self.tr_sample_size, self.te_sample_size))
        # assert self.scale == 1, "Scale (!= 1) is deprecated"        

        # 这里也改
        if self.scale != 1.:
            for i in range(len(self.all_points)):
                self.all_points[i][0] *= self.scale

        print(f"Loaded {len(self.all_points)} peptides from {root_dir}/{self.split}")

    def __len__(self):
        return len(self.all_points)

    def __getitem__(self, idx):
        points = self.all_points[idx]    # (N, 3)

        N = points.shape[0]

        if self.random_subsample:
            sampled_idxs = torch.randperm(N)[:min(N, self.tr_sample_size)]
        else:
            sampled_idxs = torch.arange(min(N, self.tr_sample_size))

        points_sampled = points[sampled_idxs, :]

        m = self.all_points_mean[idx] if self.normalize_per_shape else self.all_points_mean[0]
        s = self.all_points_std[idx] if self.normalize_per_shape else self.all_points_std[0]

        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]
        # 这里忽略了 train_points 和 test_points 的切分 为了兼容后面的训练 out 数据格式是一样的 train/test points 都是 points_sampled 
        out = {
            'idx': idx,
            'train_points': points_sampled,
            'test_points': points_sampled,  
            'mean': torch.from_numpy(m).float(),
            'std': torch.from_numpy(s).float(),
            'cate_idx': cate_idx,
            'sid': sid,
            'mid': mid
        }

        if self.use_mask:
            out['train_masks'] = torch.ones(points_sampled.shape[0], dtype=torch.bool)

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

        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size

        # peptide task 暂时只有一个类别
        self.cates = categories

        # 因为 peptide 数据都在 peptide 目录下 → synset_ids 就是 ['peptide']
        if 'all' in categories:
            self.synset_ids = ['peptide']  # 将来若多类别，可写多个
        else:
            self.synset_ids = categories   # 保留写法一致性

        # gravity_axis, display_axis_order
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

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


class PointCloudMasks(object):
    '''
    render a view then save mask
    '''
    def __init__(self, radius : float=10, elev: float =45, azim:float=315, ):

        self.radius = radius
        self.elev = elev
        self.azim = azim


    def __call__(self, points):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        camera = [self.radius * np.sin(90-self.elev) * np.cos(self.azim),
                  self.radius * np.cos(90 - self.elev),
                  self.radius * np.sin(90 - self.elev) * np.sin(self.azim),
                  ]
        # camera = [0,self.radius,0]
        _, pt_map = pcd.hidden_point_removal(camera, self.radius)

        mask = torch.zeros_like(points)
        mask[pt_map] = 1

        return mask #points[pt_map]