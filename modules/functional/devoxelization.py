from torch.autograd import Function

from modules.functional.backend import _backend

__all__ = ['trilinear_devoxelize']


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:  体素转点云
        :param coords: the coordinates of points, FloatTensor[B, 3, N] 点云中的每个点的坐标(已标准化并乘上分辨率)
        :param features: FloatTensor[B, C, R, R, R] 体素
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs, inds, wgts = _backend.trilinear_devoxelize_forward(resolution, is_training, coords, features)  # 三线性插值 找到其在体素网格中相邻的8个体素格点 计算它们的索引和对应权重 加权求和 得到该点的特征
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: 将点云特征的梯度 ∂L/∂x,根据三线性插值公式,传播回体素网格上参与插值的 8 个 voxel 上
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = _backend.trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply
