import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import open3d as o3d
import numpy as np
import torch.nn.functional as F
from easydict import EasyDict
from datasets.shapenet_data_pc import Uniform15KPC
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.transform import Rotation as R
from datasets.utils import *
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


@dataclass
class PointGrid:
    idx: int = -1
    grid_pos: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 3)))
    grid_xyz: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 3)))
    grid_pos_test: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 3)))
    grid_xyz_test: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 3)))
    grid_feats: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 3)))
    grid_feats_test: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 3)))
    cate_idx: int = -1
    sid: int = -1
    mid: int = -1

    def subset(self, mask):
        return self.__class__(
            idx=self.idx,
            grid_pos=self.grid_pos[mask],
            grid_xyz=self.grid_xyz[mask],
            grid_pos_test=self.grid_pos_test,
            grid_xyz_test=self.grid_xyz_test,
            grid_feats=self.grid_feats[mask],
            grid_feats_test=self.grid_feats_test,
            cate_idx=self.cate_idx,
            sid=self.sid,
            mid=self.mid,
        )

@dataclass
class PointCloud(ABC):
    idx: int = -1
    points: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 3)))
    feats: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 3)))
    test_points: Optional[torch.Tensor] = None      
    test_feats: Optional[torch.Tensor] = None
    mean: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    std: torch.Tensor = field(default_factory=lambda: torch.ones(3))
    cate_idx: int = -1
    sid: int = -1
    mid: int = -1
    
    def _centerlize(self):
        self.points = (self.points - self.mean)
        if self.test_points is not None:
            self.test_points = (self.test_points - self.mean)

    def _normalize(self):
        self._centerlize()
        self.points = self.points / self.std
        if self.test_points is not None:
            self.test_points = self.test_points / self.std
    
    def _random_rotate(self):
        random_quat = np.random.randn(4)
        random_quat /= np.linalg.norm(random_quat)
        rot = R.from_quat(random_quat).as_matrix()
        rot = torch.tensor(rot, dtype=self.points.dtype, device=self.points.device)
        pos_rotated = self.points @ rot.T
        assert_points_rotation_invariant_torch(self.points, pos_rotated)
        self.points = pos_rotated

        if self.test_points is not None:
            self.test_points = self.test_points @ rot.T

        self.rotation = rot.T

    def subset(self, mask):
        if self.points.shape[0] == 0:
            return self
        return self.__class__(
            points=self.points[mask],
            feats=self.feats[mask],
            idx=self.idx,
            cate_idx=self.cate_idx,
            sid=self.sid,
            mid=self.mid,
            test_points=self.test_points if self.test_points is not None else None,
            test_feats=self.test_feats if self.test_feats is not None else None,
        )

class ShapeNet15kPointCloudTrees(Uniform15KPC):
    def __init__(self, root_dir="data/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], 
                 tr_sample_size=10000, 
                 te_sample_size=2048,
                 split='train', 
                 scale=1., 
                 normalize_per_shape=False,
                 normalize_std_per_axis=False, 
                 box_per_shape=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None,
                 use_mask=False,
                 merge_level=7,
                 random_rotation=True,
                 grid_len=0.05,
                 xyz_resolution=0.005,
                 enable_cutoff=False):
        
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]
        self.merge_level = merge_level
        self.random_rotation = random_rotation and split == 'train'
        self.grid_len = grid_len
        self.xyz_resolution = xyz_resolution
        self.num_xyz = round(self.grid_len / self.xyz_resolution)
        self.enable_cutoff = enable_cutoff
        super(ShapeNet15kPointCloudTrees, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape, 
            box_per_shape=box_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, 
            all_points_std=all_points_std,
            input_dim=3, use_mask=use_mask)
    
    def make_grid(self, point_cloud: PointCloud):
        point_cloud._centerlize()
        if self.random_rotation:
            point_cloud._random_rotate()
        max_grid_size = 2**self.merge_level

        def process(point_cloud: PointCloud, test_points=False):
            if not test_points and point_cloud.test_points is not None:
                process_points = point_cloud.points
            else:
                process_points = point_cloud.test_points
            center = process_points.mean(axis=0)
            process_points = process_points - center + self.grid_len / 2.0
            
            grid_pos, grid_xyz = self.phy_pos_to_grid_pos(process_points)
            grid_pos = grid_pos - torch.min(grid_pos, dim=0, keepdims=True).values

            assert (grid_pos >= 0).all(), grid_pos
            assert (grid_xyz >= 0).all() and (
                grid_xyz < self.num_xyz
            ).all(), grid_xyz

            grid_size = torch.tensor([max_grid_size] * 3, dtype=torch.long)
            mean_grid_pos = (
                grid_pos.float().mean(dim=0).long()
            )
            # place to the center
            grid_pos = grid_pos - mean_grid_pos + grid_size // 2
            # TODO: random shift the center

            return grid_pos, grid_xyz
        
        grid_pos, grid_xyz = process(point_cloud, test_points=False)
        grid_pos_test, grid_xyz_test = process(point_cloud, test_points=True)
        
        cutoff_count = 0
        is_exceed = grid_pos.min() < 0 or grid_pos.max() >= max_grid_size
        enable_cutoff = self.enable_cutoff
        if is_exceed and enable_cutoff:
            # perform cutoff, by random placing a grid_size * grid_size * grid_size grid into the current grid
            new_point_cloud, cutoff_count = self.cutoff_by_points(
                point_cloud, max_grid_size
            )
            assert new_point_cloud.pos.shape[0] > 0
            grid_pos, grid_xyz = process(new_point_cloud)
            grid_pos_test, grid_xyz_test = process(new_point_cloud, test_points=True)
            point_cloud = new_point_cloud

        assert (
            grid_pos.min() >= 0 and grid_pos.max() < max_grid_size
        ), "grid pos exceed max size, please use larger merge_level, or enable cutoff"
        return grid_pos, grid_xyz, grid_pos_test, grid_xyz_test, point_cloud, cutoff_count
    
    def phy_pos_to_grid_pos(self, phy_pos):
        grid_pos = torch.floor(phy_pos / self.grid_len).long()
        grid_xyz = torch.round(
            (phy_pos % self.grid_len) / self.xyz_resolution
        ).long()
        max_xyz = self.num_xyz
        overflow_mask = grid_xyz >= max_xyz
        grid_pos += overflow_mask
        grid_xyz[overflow_mask] = 0
        return grid_pos, grid_xyz

    def cutoff_by_points(self, point_cloud, grid_size):
        select_points = np.random.choice(point_cloud.points.shape[0], 1, replace=False)[0]
        select_c_pos = point_cloud.points[select_points]
        axis_min_t = select_c_pos - grid_size * self.grid_len // 2
        axis_max_t = select_c_pos + (grid_size - 1) * self.grid_len // 2
        pos_flag = torch.all(
            (point_cloud.points > axis_min_t) & (point_cloud.points < axis_max_t),
            axis=1,
        )
        return point_cloud.subset(pos_flag), (pos_flag == False).sum()


    def construct_tree_sequence(self, point_grid: PointGrid):
        max_grid_size = 2**self.merge_level
        grid_size = torch.tensor([max_grid_size] * 3, dtype=torch.long)
        grid_pos = point_grid.grid_pos
        # by default, each pos should have only one atom
        point_grid.grid_feats = point_grid.grid_xyz.float()
        if point_grid.grid_feats is None:
            unique_indices = sample_from_duplicate_torch(grid_pos)
            point_grid = point_grid.subset(unique_indices)
        else:
            unique_indices, reduced_grid_feat = average_feats_by_grid_pos(grid_pos, point_grid.grid_feats)
            point_grid = point_grid.subset(unique_indices)
            point_grid.grid_feats = reduced_grid_feat
        
        grid_pos = point_grid.grid_pos
        (
            tree_pos,
            tree_type,
            tree_phy_pos,
            tree_count,
        ) = self.get_tree_feat(atom_grid_pos)


    def __getitem__(self, idx):
        out = super(ShapeNet15kPointCloudTrees, self).__getitem__(idx)
        point_cloud = {
            'idx': out['idx'],
            'points': out['train_points'],
            'feats': None,
            'cate_idx': out['cate_idx'],
            'sid': out['sid'],
            'mid': out['mid'],
            'mean': out['mean'],
            'std': out['std'],
            'test_points': out['test_points'],
            'test_feats': None,
        }
        point_cloud = PointCloud(**point_cloud)
        (grid_pos, grid_xyz, grid_pos_test, grid_xyz_test, 
         point_cloud, cutoff_count) = self.make_grid(point_cloud)
        
        point_grid = PointGrid(point_cloud.idx,
                               grid_pos, 
                               grid_xyz, 
                               grid_pos_test, 
                               grid_xyz_test, 
                               point_cloud.feats, 
                               point_cloud.test_feats, 
                               point_cloud.cate_idx, 
                               point_cloud.sid, 
                               point_cloud.mid)
        point_grid = self.construct_tree_sequence(point_grid)

        return out



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


