import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import open3d as o3d
import numpy as np
import torch.nn.functional as F

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15KPC(Dataset):
    def __init__(self, root_dir, subdirs, tr_sample_size=10000,
                 te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, box_per_shape=False,
                 random_subsample=False,
                 normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None,
                 input_dim=3, use_mask=False):
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size  
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale  # 点云的缩放比例
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.use_mask = use_mask
        self.box_per_shape = box_per_shape
        if use_mask:
            self.mask_transform = PointCloudMasks(radius=5, elev=5, azim=90)

        self.all_cate_mids = []  # 所有点云数据对应的 (synset id, model id) 列表 ("02691156", "train/1a9b552befd6306cc8f2d5fe7449af61")
        self.cate_idx_lst = []  # 每个样本对应的类别索引
        self.all_points = []  # 所有点云数据本身
        for cate_idx, subd in enumerate(self.subdirs):  # 遍历类别
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)  # 拼接类别路径
            if not os.path.isdir(sub_path):  # 检查路径是否存在
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []  # 找到该类别下所有.npy 文件的模型 id   一个类别的 train/test/val 下面有许多.npy 文件，文件名就是 id   
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:  # 逐个加载 npy 文件
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root_dir, subd, mid + ".npy")  # 拼成路径
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)  # 读取点云

                except:
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)  固定打乱顺序
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization  四种归一化方式
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        elif self.box_per_shape:
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.min(axis=1).reshape(B, 1, input_dim)
            self.all_points_std = self.all_points.max(axis=1).reshape(B, 1, input_dim) - self.all_points.min(axis=1).reshape(B, 1, input_dim)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)  # [1,1,3]
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)  # [1,1,1]

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std  # normalize 所有点
        if self.box_per_shape:
            self.all_points = self.all_points - 0.5
        self.train_points = self.all_points[:, :10000]  # 切分训练集 测试集
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)  # 限制采样大小
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape or self.box_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s


        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        out = {
            'idx': idx,  # 索引号
            'train_points': tr_out,  # (tr_sample_size, 3)
            'test_points': te_out,  # (te_sample_size, 3)
            'mean': m, 'std': s, 'cate_idx': cate_idx,  # mean std用于把生成点云还原到原始坐标空间 中心对齐  cate_idx 是类别索引 如 airplane=0, bag=1, basket=2
            'sid': sid, 'mid': mid  # sid 是类别 synset id 如 '02691156'，mid 是模型 id 如 'train/1a9b552befd6306cc8f2d5fe7449af61'
        }

        if self.use_mask:
            tr_mask = self.mask_transform(tr_out)
            out['train_masks'] = tr_mask

        return out


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False, box_per_shape=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None,
                 use_mask=False):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size  # 指定每个样本点云内选多少个点 10000
        self.te_sample_size = te_sample_size  # 指定每个样本点云内选多少个点 2048
        self.cates = categories  # 类别列表 如["chair", "table"]
        if 'all' in categories:  # 如果全类别 就直接使用所有类别
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]  # 否则找对应的 比如chair=02691156

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1  # 重力轴 y 轴 用于可视化的时候选取
        self.display_axis_order = [0, 2, 1]  # 可视化的时候选取的轴顺序

        super(ShapeNet15kPointClouds, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,  # 是否对每个 shape 分别做 normalization
            box_per_shape=box_per_shape,  # 是否对每个 shape 做 box normalization
            normalize_std_per_axis=normalize_std_per_axis,  # 是否对每个 shape 的每个轴分别做标准化
            random_subsample=random_subsample,  # 是否随机子采样
            all_points_mean=all_points_mean,  # 全部点云的均值
            all_points_std=all_points_std,  # 全部点云的标准差
            input_dim=3, use_mask=use_mask)



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


####################################################################################


