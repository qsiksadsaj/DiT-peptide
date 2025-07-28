# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import os
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from modules.voxelization import Voxelization
import modules.functional as F

import numpy as np
import matplotlib.pyplot as plt

def visualize_voxel_grid(voxel_grid, save_path="/root/DiT-3D/temp/voxel_vision/voxel.png", threshold=1e-3):
    """
    voxel_grid: [C, R, R, R] numpy array
    save_path: str, path to save figure
    """
    if voxel_grid.ndim == 4:
        voxel_grid = voxel_grid.sum(axis=0)

    coords = np.argwhere(voxel_grid > threshold)
    values = voxel_grid[voxel_grid > threshold]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        coords[:,0],
        coords[:,1],
        coords[:,2],
        c=values,
        cmap='viridis',
        s=30
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Voxel Grid Visualization")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    # print(f"✅ Voxel grid visualization saved to {save_path}")



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class PatchEmbed_Voxel(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=32, patch_size=4, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        voxel_size = (voxel_size, voxel_size, voxel_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    # def forward(self, x):
    #     B, C, X, Y, Z = x.shape
    #     x = x.float()
    #     if torch.isnan(x).any():
    #         print("NaN detected in x")
    #     x = self.proj(x).flatten(2).transpose(1, 2)  # 切分小体素块 转 emb 展平 转置
    #     if torch.isnan(x).any():
    #         print("NaN detected in x")
    #     return x

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = x.float()

        if torch.isnan(x).any():
            print("[Pre-proj] NaN detected in input x")
        x = x.clamp(min=-1.0, max=1.0)  # 或者根据实际 max 值稍微放宽一点

        x_proj = self.proj(x)  # [B, embed_dim, x/p, y/p, z/p]

        if torch.isnan(x_proj).any():
            print("[Post-proj] NaN detected in self.proj(x)")

        x_flat = x_proj.flatten(2)  # [B, embed_dim, N_patch]
        x_trans = x_flat.transpose(1, 2)  # [B, N_patch, embed_dim]

        if torch.isnan(x_trans).any():
            print("[Post-transpose] NaN detected in patch embeddings")

        return x_trans

        

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # print('token drop drop_ids:', drop_ids, drop_ids.shape)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        # print('token drop labels:', labels, labels.shape)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        approx_gelu = lambda: nn.GELU() # for torch 1.7.1
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        '''
        给 Attention 分支生成：

        shift_msa:均值偏移(用于 LayerNorm)
        scale_msa:尺度缩放(用于 LayerNorm)
        gate_msa:残差路径门控(Residual Gating)

        给 MLP 分支生成：

        shift_mlp:均值偏移
        scale_mlp:尺度缩放
        gate_mlp:残差路径门控
        '''
        self.adaLN_modulation = nn.Sequential(  
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        if torch.isnan(shift_msa).any() or torch.isnan(scale_msa).any() or torch.isnan(gate_msa).any() or torch.isnan(shift_mlp).any() or torch.isnan(scale_mlp).any() or torch.isnan(gate_mlp).any():
            print("NaN detected in x")  
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if torch.isnan(x).any():
            print("NaN detected in x")
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if torch.isnan(x).any():
            print("NaN detected in x")
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # 重建成[out_channels, patch_size, patch_size, patch_size]的体素块
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=16,
        patch_size=4,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma  # 决定是只预测噪声 还是同时预测噪声和方差
        self.in_channels = in_channels  # 输入通道数 点云任务中常为 3
        self.out_channels = in_channels * 2 if learn_sigma else in_channels  # 输出通道数 点云任务中常为 3 或 6
        self.patch_size = patch_size  # CV_Transformer 都要做 patch  把输入点云划分成 patch 当做 token 输入
        self.num_heads = num_heads  # 多头注意力头数

        self.input_size = input_size  # 输入点云的分辨率
        self.voxelization = Voxelization(resolution=input_size, normalize=True, eps=0)  # 体素化模块

        self.x_embedder = PatchEmbed_Voxel(input_size, patch_size, in_channels, hidden_size, bias=True)  # 点云的Patch嵌入
        num_patches = self.x_embedder.num_patches
        
        self.t_embedder = TimestepEmbedder(hidden_size)  # 时间步嵌入
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)  # 类别标签嵌入

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)  # 位置嵌入

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)  # Dit 模块
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.input_size//self.patch_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    # raw
    def unpatchify_voxels(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)    (N, 64, 8*8*8*3)
        voxels: (N, C, X, Y, Z)          (N, 3, 32, 32, 32)
        """
        c = self.out_channels
        p = self.patch_size
        x = y = z = self.input_size // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)  # 把patch grid的结构重排为 voxel grid 的结构
        points = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return points

    # raw
    def forward(self, x, t, y, mask=None):
        """
        Forward pass of DiT.
        x: (N, C, P) tensor of spatial inputs (point clouds or latent representations of images)  输入的点云
        t: (N,) tensor of diffusion timesteps  每个样本的扩散时间步
        y: (N,) tensor of class labels  每个样本的类别标签（条件扩散）
        mask: (N, P) tensor of boolean values indicating valid points in the point cloud  每个样本的点云是否有效的掩码
        """
        # xj_mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)        # (B, 1, N)
            x = x * mask_expanded

        # Voxelization
        features, coords = x, x
        x, voxel_coords = self.voxelization(features, coords)  # 点云转体素
        if torch.isnan(x).any():
            print("NaN detected in x")
        if torch.isnan(voxel_coords).any():
            print("NaN detected in voxel_coords")
        # # 可视化 voxif torch.isnan(x).any():
        # if self.training:
        #     voxel_grid = x[0].detach().cpu().numpy()
        #     visualize_voxel_grid(voxel_grid, threshold=1e-3)

        x = self.x_embedder(x)   # 将体素网格划分成patch并flatten成token
        x = x + self.pos_embed  # 添加位置嵌入
        if torch.isnan(x).any():
            print("NaN detected in x")
        t = self.t_embedder(t)  # 添加时间步嵌入
        y = self.y_embedder(y, self.training)  # 添加类别标签嵌入
        c = t + y                                

        for block in self.blocks:
            x = block(x, c)  # Dit模块            
            if torch.isnan(x).any():
                print("NaN detected in x")
                raise ValueError("NaN detected in DiT block output.")          
        x = self.final_layer(x, c)  # 将输出还原为patch格式           
        if torch.isnan(x).any():
            print("NaN detected in x")     
        x = self.unpatchify_voxels(x)  # 将patch恢复为完整的体素体            
        if torch.isnan(x).any():
                    print("NaN detected in x")
        # Devoxelization
        x = F.trilinear_devoxelize(x, voxel_coords, self.input_size, self.training)
        if torch.isnan(x).any():
                    print("NaN detected in x")
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    print('grid_size:', grid_size)

    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(pretrained=False, **kwargs):

    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load('/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if not k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model

def DiT_XL_4(pretrained=False, **kwargs):

    model = DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load('/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)
    
    return model

def DiT_XL_8(pretrained=False, **kwargs):

    model = DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load('/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if not k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model

def DiT_L_2(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_B_16(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=16, num_heads=12, **kwargs)

def DiT_B_32(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=32, num_heads=12, **kwargs)

def DiT_S_2(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(pretrained=False, **kwargs):

    model = DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)
    if pretrained:
        checkpoint = torch.load('/path/to/DiT2D_pretrained_weights/DiT-S-4.pth', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model

def DiT_S_8(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_S_16(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=16, num_heads=6, **kwargs)

def DiT_S_32(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=32, num_heads=6, **kwargs)

DiT3D_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}