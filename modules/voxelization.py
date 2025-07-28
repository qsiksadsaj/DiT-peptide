import torch
import torch.nn as nn

import modules.functional as F

__all__ = ['Voxelization']

# 如果输入的点云十分稠密  就可以用这种voxel构造方式。这里点聚集平均到一个 voxel 格子里  一个点不会影响周围的 voxel 格子；但如果输入点云太稀疏  形成的 voxel 也会太稀疏
class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0, scale=1.0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps
        self.scale = scale

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)  # 中心化处理
        if self.normalize:  # 标准化
            # norm_coords = norm_coords / (self.scale + self.eps) / 2.0 + 0.5

            denom = norm_coords.norm(dim=1,  keepdim=True).max(dim=2, keepdim=True).values 
            denom = torch.clamp(denom  * 2.0, min=1e-6)  # 硬性下限
            norm_coords = norm_coords / denom + 0.5 
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)  # 转为voxel坐标索引 coord x resolution 再 round()成 voxel 中的整数索引 
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords  # 将点聚合为体素体

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')

# class Voxelization(nn.Module):
#     def __init__(self, resolution, normalize=True, eps=0, scale=1.0, range_n=2.5):
#         super().__init__()
#         self.r = int(resolution)
#         self.normalize = normalize
#         self.eps = eps
#         self.scale = scale
#         self.range_n = range_n  # 新增参数：坐标范围 [-n, n]

#     def forward(self, features, coords):
#         coords = coords.detach()
#         norm_coords = coords - coords.mean(2, keepdim=True)

#         if self.normalize:
#             denom = norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values
#             denom = torch.clamp(denom, min=1e-6)
#             norm_coords = norm_coords / denom * self.range_n
#         else:
#             norm_coords = torch.clamp(norm_coords, -self.range_n, self.range_n)

#         # 映射到 voxel 索引
#         scaled_coords = (norm_coords + self.range_n) / (2 * self.range_n)
#         scaled_coords = torch.clamp(scaled_coords * self.r, 0, self.r - 1)
#         vox_coords = torch.round(scaled_coords).to(torch.int32)

#         return F.avg_voxelize(features, vox_coords, self.r), norm_coords

#     def extra_repr(self):
#         return f'resolution={self.r}, normalize={self.normalize}, range_n={self.range_n}'


# import torch
# import torch.nn as nn

# class Voxelization(nn.Module):
#     """
#     高斯加权体素化模块：适用于稀疏点云，每个点对周围体素区域进行高斯加权。
#     """

#     def __init__(self, resolution, normalize=True, eps=1e-6, radius=2, sigma=0.8):
#         """
#         Args:
#             resolution (int): voxel 网格的边长（例如 64 表示 64x64x64）
#             normalize (bool): 是否对输入坐标进行归一化
#             eps (float): 防止除零的小常数
#             radius (int): 每个点的高斯影响范围（单位为 voxel 格子数）
#             sigma (float): 高斯标准差，控制扩散程度
#         """
#         super().__init__()
#         self.r = int(resolution)
#         self.normalize = normalize
#         self.eps = eps
#         self.radius = radius
#         self.sigma = sigma

#     def forward(self, features, coords):
#         """
#         Args:
#             features: [B, C, N] 点的特征（例如坐标值，或统一为1）
#             coords: [B, 3, N] 点的坐标，范围任意（通常为 [-1, 1] 或实际坐标）
#         Returns:
#             voxel_grid: [B, C, R, R, R] 加权体素张量
#             coords_voxel: [B, 3, N] 归一化并缩放到体素网格的 voxel 坐标
#         """
#         B, C, N = features.shape
#         coords = coords.detach()

#         # === 坐标中心化 & 归一化到 [0,1]
#         norm_coords = coords - coords.mean(dim=2, keepdim=True)
#         # 检查是否全为 0
#         mask_nan = (
#             torch.isnan(norm_coords).any(dim=1).any(dim=1) | 
#             (norm_coords.abs().sum(dim=(1, 2)) < 1e-6)
#         )
#         if mask_nan.any():
#             print("❗ Warning: Detected degenerate input samples (NaN or zero coords), skipping them.")
#             norm_coords[mask_nan] = 0.5  # 设置为中间 voxel
#             # 或者 raise / continue 处理
#         if self.normalize:
#             denom = norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values
#             denom = torch.clamp(denom * 2.0, min=self.eps)
#             norm_coords = norm_coords / denom * 0.99 + 0.5
#             # === 检查是否出现 NaN ===
#             if not torch.isfinite(norm_coords).all():
#                 print("❗ NaN in normalized coordinates!")
#                 print("Original coords stats:", coords.mean().item(), coords.std().item())
#                 print("Denom:", denom.flatten())
#                 raise RuntimeError("NaN in normalized coords after division.")
#         else:
#             norm_coords = (norm_coords + 1) / 2.0

#         # === 缩放到 voxel 网格
#         norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
#         coords_voxel = norm_coords

#         # === 初始化 voxel 网格
#         voxel_grid = torch.zeros((B, C, self.r, self.r, self.r), device=features.device)
#         weight_sum = torch.zeros((B, 1, self.r, self.r, self.r), device=features.device)

#         sigma2 = self.sigma ** 2

#         for b in range(B):
#             for n in range(N):
#                 x_f, y_f, z_f = coords_voxel[b, :, n]  # float
#                 x0, y0, z0 = int(x_f), int(y_f), int(z_f)

#                 for dx in range(-self.radius, self.radius + 1):
#                     for dy in range(-self.radius, self.radius + 1):
#                         for dz in range(-self.radius, self.radius + 1):
#                             xi, yi, zi = x0 + dx, y0 + dy, z0 + dz
#                             if 0 <= xi < self.r and 0 <= yi < self.r and 0 <= zi < self.r:
#                                 dist2 = (x_f - xi) ** 2 + (y_f - yi) ** 2 + (z_f - zi) ** 2
#                                 weight = torch.exp(-dist2 / (2 * sigma2))
#                                 voxel_grid[b, :, xi, yi, zi] += weight * features[b, :, n]
#                                 weight_sum[b, :, xi, yi, zi] += weight

#         # === 加权平均 & NaN 防御
#         voxel_grid = voxel_grid / (weight_sum + self.eps)
#         valid_mask = (weight_sum > 1e-3).float()
#         voxel_grid = voxel_grid * valid_mask

#         # === Debug 检查 NaN
#         if torch.isnan(voxel_grid).any():
#             print("❗ NaN detected in voxelization output")
#             raise RuntimeError("NaN in voxel grid")

#         return voxel_grid, coords_voxel
