import os
import torch

from pprint import pprint
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, compute_rmsd_peptide
from metrics.evaluation_metrics_peptide import compute_all_metrics_peptide

import torch.nn as nn
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from tqdm import tqdm

from datasets.peptide_data import PeptidePointClouds
from models.dit3d import DiT3D_models
from utils.misc import Evaluator

from rmsd import *

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, y, clip_denoised: bool, return_pred_xstart: bool):
        
        model_output = denoise_fn(data, t, y)  # 这里 denoise_fn 是模型的前向传播函数

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, y, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, y=y, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device, y, 
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def reconstruct(self, x0, t, y, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            # img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn, y=y,
            #                       clip_denoised=False, return_pred_xstart=False, use_var=True).detach()
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=False, return_pred_xstart=False).detach()  # xj


        return img_t


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)
        
        # DiT-3d
        self.model = DiT3D_models[args.model_type](input_size=args.voxel_size, num_classes=args.num_classes)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, y, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, y, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, y, mask=None):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, y, mask)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, mask, noises=None, y=None):
        B, D, N = data.shape                           # [16, 3, 2048]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, y=y, mask=mask)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, y, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas

def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    '''

    :param target_shape_constraint: target voxels
    :return: constrained x
    '''
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000)**2 ))
    def constrain_fn(x, t):
        eps_ =  eps_all[t] if (t<1000) else 0
        for _ in range(num_steps):
            x  = x - eps_ * ((x - ground_truth) * mask)


        return x
    return constrain_fn


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


#############################################################################

def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = PeptidePointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=True,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = PeptidePointClouds(root_dir=dataroot,
        categories=[category], split='test',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=True,
        normalize_std_per_axis=False,
        random_subsample=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None, collate_fn=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True, collate_fn=collate_fn)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False, collate_fn=collate_fn)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler

def peptide_collate_fn(batch):
    out = {}

    # === train_points
    train_points_list = []
    for b in batch:
        pts = b["train_points"]
        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts).float()
        train_points_list.append(pts)
    train_points_padded = pad_sequence(
        train_points_list, batch_first=True, padding_value=0.0
    )
    out["train_points"] = train_points_padded

    # === test_points
    test_points_list = []
    for b in batch:
        pts = b["test_points"]
        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts).float()
        test_points_list.append(pts)
    test_points_padded = pad_sequence(
        test_points_list, batch_first=True, padding_value=0.0
    )
    out["test_points"] = test_points_padded

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
        if key in ["train_points", "test_points"]:
            continue

        values = [b[key] for b in batch]

        if isinstance(values[0], np.ndarray):
            out[key] = torch.stack([torch.from_numpy(v).float() for v in values])
        elif isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            out[key] = torch.tensor(values)
        else:
            out[key] = values

    return out


# 生成的时候设置 bs=1  不用考虑 mask 的问题
def generate_peptide_eval(model, opt, gpu, outf_syn, evaluator):

    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)

    _, test_dataloader, _, test_sampler = get_dataloader(
        opt,
        test_dataset,
        test_dataset,
        collate_fn=peptide_collate_fn
    )

    def new_y_chain(device, num_chain):
        # peptide 全部是 0 类  如果扩充成 环肽 普通多肽  这里上条件类别
        return torch.zeros(num_chain, dtype=torch.long, device=device)

    with torch.no_grad():

        samples_list = []

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):

            x = data['test_points'].transpose(1,2)
            m = data['mean'].float()
            s = data['std'].float()
            y = data['cate_idx']

            gen = model.gen_samples(
                shape=x.shape,
                device=gpu,
                y=new_y_chain(gpu, y.shape[0]),
                clip_denoised=False
            ).detach().cpu()

            gen = gen.transpose(1,2).contiguous()
            x = x.transpose(1,2).contiguous()
            # print("gen range before:", gen.min().item(), gen.max().item())
            # print("x range before:", x.min().item(), x.max().item())
            gen = gen * s + m
            x = x * s + m

            mid = data["mid"][0]

            # === 保存到以样本ID命名的子目录 ===
            save_root = "./debug_test_reconstruction_19999"  # 顶级输出目录
            sample_id = mid.replace("/", "_")          # e.g., train_7qs9_e

            sample_dir = os.path.join(save_root, sample_id)
            os.makedirs(sample_dir, exist_ok=True)

            gt_path = os.path.join(sample_dir, "gt.xyz")
            pred_path = os.path.join(sample_dir, "pred.xyz")

            np.savetxt(gt_path, x[0].cpu().numpy(), fmt="%.4f")
            np.savetxt(pred_path, gen[0].cpu().numpy(), fmt="%.4f")

            # print("gen range after:", gen.min().item(), gen.max().item())
            # print("x range after:", x.min().item(), x.max().item())

            # samples_list.append(gen.cuda())
            samples_list.append({
                "mid": mid,
                "gen_coords": gen[0].cpu()
            })

            # visualize_pointcloud_batch(
            #     os.path.join(outf_syn, f"{i}_{gpu}.png"),
            #     gen,
            #     None, None, None
            # )

            # Compute metrics
            # results = compute_all_metrics(gen,x,opt.bs)  # 这是多 bs 的指标
            results = compute_rmsd_peptide(gen, x)  # 这是单 bs 的指标

            # detach & cpu 转 float
            results = {
                k: (v.cpu().detach().item() if not isinstance(v, float) else v)
                for k, v in results.items()
            }

            # JSD
            jsd = JSD(gen.numpy(), x.numpy())
            evaluator.update(results, jsd)


        stats = evaluator.finalize_stats()
        torch.save(samples_list, opt.eval_path)
    
    # === Step 1: 收集所有预测和真实点云 ===
    gen_all = []
    gt_all = []

    for s in samples_list:
        gen_all.append(s["gen_coords"])  # Tensor (N_i, 3)
        mid = s["mid"]
        gt_path = os.path.join("./debug_test_reconstruction_19999", mid.replace("/", "_"), "gt.xyz")
        gt_coords = torch.tensor(np.loadtxt(gt_path)).float()
        gt_all.append(gt_coords)

    # === Step 2: padding → 拼接为 batch ===
    def pad_and_stack(point_list):
        max_len = max(p.shape[0] for p in point_list)
        padded = []
        for p in point_list:
            pad_len = max_len - p.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, 3)
                p = torch.cat([p, pad], dim=0)
            padded.append(p)
        return torch.stack(padded)  # (B, N, 3)

    gen_all = pad_and_stack(gen_all)          # (B, N, 3)
    gt_all = pad_and_stack(gt_all)            # (B, N, 3)

    # === Step 2.5: 构造 mask（因为 padding 了）===
    sample_masks = (gen_all.abs().sum(dim=1) > 0)  # (B, N) 有效点为 True
    ref_masks = (gt_all.abs().sum(dim=1) > 0)      # (B, N)

    # === Step 3: 转为 (B, 3, N) 格式（符合 metric 输入）===
    gen_all = gen_all.permute(0, 2, 1).contiguous()  # (B, 3, N)
    gt_all = gt_all.permute(0, 2, 1).contiguous()    # (B, 3, N)

    # === Step 4: 计算整体评估指标 ===
    metrics = compute_all_metrics_peptide(
        gen_all, gt_all,
        batch_size=opt.bs
    )
    metrics = {k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
    jsd = JSD(gen_all.permute(0, 2, 1).numpy(), gt_all.permute(0, 2, 1).numpy())

    # === Step 5: 打印 ===
    print("========== Overall Evaluation Metrics ==========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"JSD: {jsd:.4f}")


    return stats

@torch.no_grad()
def forward_denoise_eval(model, opt, gpu, outf_syn, evaluator):
    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)
    _, test_dataloader, _, test_sampler = get_dataloader(
        opt,
        test_dataset,
        test_dataset,
        collate_fn=peptide_collate_fn
    )

    samples_list = []

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Forward Denoising'):
        x = data['test_points'].transpose(1,2).to(gpu)     # shape (B, 3, N)
        m = data['mean'].float().to(gpu)
        s = data['std'].float().to(gpu)
        y = data['cate_idx'].to(gpu)

        # === 模型预测 ===
        t = torch.zeros(x.shape[0], dtype=torch.long, device=gpu)
        out = model._denoise(x, t, y).detach()  # shape (B, 3, N)

        # === 还原尺度 ===
        gen = out.transpose(1,2) * s + m  # shape (B, N, 3)
        gt = x.transpose(1,2) * s + m

        mid = data["mid"][0]
        sample_dir = os.path.join("./debug_test_denoise", mid.replace("/", "_"))
        os.makedirs(sample_dir, exist_ok=True)

        np.savetxt(os.path.join(sample_dir, "gt.xyz"), gt[0].cpu().numpy(), fmt="%.4f")
        np.savetxt(os.path.join(sample_dir, "pred.xyz"), gen[0].cpu().numpy(), fmt="%.4f")

        samples_list.append({
            "mid": mid,
            "gen_coords": gen[0].cpu()
        })

        results = compute_rmsd_peptide(gen, gt)
        results = {k: (v.cpu().item() if not isinstance(v, float) else v) for k, v in results.items()}
        jsd = JSD(gen.cpu().numpy(), gt.cpu().numpy())
        evaluator.update(results, jsd)

    torch.save(samples_list, opt.eval_path)
    return evaluator.finalize_stats()

# # 不带scale还原的
# @torch.no_grad()
# def forward_rmsd_eval(model, opt, gpu, output_dir):
#     _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)
#     _, test_dataloader, _, _ = get_dataloader(opt, test_dataset, test_dataset, collate_fn=peptide_collate_fn)

#     save_root = os.path.join(output_dir, "denoise_rmsd_results")
#     os.makedirs(save_root, exist_ok=True)

#     rmsds = []

#     for i, data in enumerate(tqdm(test_dataloader, desc="RMSD Denoising")):
#         x0 = data["test_points"].to(gpu).transpose(1, 2)   # (1, 3, N)
#         mask = data["test_masks"].to(gpu)                  # (1, N)
#         y = data["cate_idx"].to(gpu)

#         # Step 1: 时间步 t（你可以设定固定值）
#         t = torch.randint(low=int(opt.time_num * 0.8), high=opt.time_num, size=(1,), device=gpu)

#         # Step 2: 添加噪声
#         x_t = model.diffusion.q_sample(x_start=x0, t=t)

#         # Step 3: 去噪重构
#         x_pred = model.diffusion.reconstruct(x0, t.item(), y, model._denoise)  # (1, 3, N)

#         # Step 4: RMSD
#         x0_masked = x0[0, :, mask[0]].transpose(0, 1).cpu()       # (N_i, 3)
#         x_pred_masked = x_pred[0, :, mask[0]].transpose(0, 1).cpu()  # (N_i, 3)

#         rmsd_result = compute_rmsd_peptide(x_pred_masked.unsqueeze(0), x0_masked.unsqueeze(0))
#         rmsds.append(rmsd_result["rmsd"].item())

#         # Step 5: 可选保存
#         mid = data["mid"][0].replace("/", "_")
#         sample_dir = os.path.join(save_root, mid)
#         os.makedirs(sample_dir, exist_ok=True)
#         np.savetxt(os.path.join(sample_dir, "gt.xyz"), x0_masked.numpy(), fmt="%.4f")
#         np.savetxt(os.path.join(sample_dir, "pred.xyz"), x_pred_masked.numpy(), fmt="%.4f")

#     # Step 6: 汇总
#     avg_rmsd = sum(rmsds) / len(rmsds)
#     print(f"\n======== Denoising RMSD Evaluation ========")
#     print(f"Average RMSD over {len(rmsds)} samples: {avg_rmsd:.4f}")


# 带scale还原
@torch.no_grad()
def forward_rmsd_eval(model, opt, gpu, output_dir):
    '''你可以根据自己的配置调整路径'''
    # meta_dict_path = os.path.join(opt.dataroot, "peptide", "scale_center_test.npy")  # dataroot这块决定用哪个缩放的结果
    meta_dict_path = "/root/DiT-3D/data/peptides_only_ligand_raw_no_norm/peptide/scale_center_test.npy"
    raw_pdb_root = os.path.join(opt.dataroot.replace("only_ligand_raw", "dataset_xj"), "PPDbench_xj_10_repaired")  

    meta_dict = np.load(meta_dict_path, allow_pickle=True).item()

    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)
    _, test_dataloader, _, _ = get_dataloader(opt, test_dataset, test_dataset, collate_fn=peptide_collate_fn)

    save_root = os.path.join(output_dir, "denoise_rmsd_results")
    os.makedirs(save_root, exist_ok=True)

    rmsds = []
    skipped = 0

    for i, data in enumerate(tqdm(test_dataloader, desc="RMSD Denoising")):
        x0 = data["test_points"].to(gpu).transpose(1, 2)   # (1, 3, N)
        mask = data["test_masks"].to(gpu)
        y = data["cate_idx"].to(gpu)
        mid = data["mid"][0]
        sample_id = mid.split("/")[-1]

        if sample_id not in meta_dict:
            print(f"[{i:03d}] {sample_id} not in scale_center中")
            skipped += 1
            continue

        # Step 1: 添加噪声 + 去噪
        t = torch.randint(low=int(opt.time_num * 0.8), high=opt.time_num, size=(1,), device=gpu)
        x_pred = model.diffusion.reconstruct(x0, t.item(), y, model._denoise)  # (1, 3, N)

        # Step 2: 反归一化
        gen = x_pred[0, :, mask[0]].transpose(0, 1).cpu().numpy()  # (N, 3)
        scale = meta_dict[sample_id]["scale"]
        center = np.array(meta_dict[sample_id]["center"])
        gen_denorm = gen * scale + center

        # Step 3: 获取真实结构
        pdb_path = os.path.join(raw_pdb_root, sample_id, "peptide_repaired.pdb")
        if not os.path.exists(pdb_path):
            print(f"[{i:03d}] PDB not found:{pdb_path}")
            skipped += 1
            continue
        gt = pdb_to_ca_point_cloud(pdb_path)

        # Step 4: 对齐 + RMSD
        if gt.shape != gen_denorm.shape:
            print(f"[{i:03d}] ⚠️ GT/GEN 点数不一致: {gt.shape} vs {gen_denorm.shape}")
            skipped += 1
            continue

        gen_aligned = kabsch_align(gen_denorm, gt)
        rmsd = compute_rmsd(gen_aligned, gt)
        rmsds.append(rmsd)

        # Step 5: 可选写出对比
        sample_dir = os.path.join(save_root, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        np.savetxt(os.path.join(sample_dir, "pred.xyz"), gen_denorm, fmt="%.3f")
        np.savetxt(os.path.join(sample_dir, "gt.xyz"), gt, fmt="%.3f")

        # print(f"[{i:03d}] ✅ RMSD = {rmsd:.3f} Å")

    # Step 6: 汇总输出
    if len(rmsds) > 0:
        rmsd_array = np.array(rmsds)
        print("\n======== Denoising RMSD Evaluation ========")
        print(f"样本数量: {len(rmsd_array)} / 总数: {len(rmsd_array) + skipped}")
        print(f"均值 (Mean RMSD): {rmsd_array.mean():.3f} Å")
        print(f"标准差 (Std): {rmsd_array.std():.3f} Å")
        print(f"最小值 (Min): {rmsd_array.min():.3f} Å")
        print(f"最大值 (Max): {rmsd_array.max():.3f} Å")
        print(f"中位数 (Median): {np.median(rmsd_array):.3f} Å")
        print(f"25% 分位数: {np.percentile(rmsd_array, 25):.3f} Å")
        print(f"75% 分位数: {np.percentile(rmsd_array, 75):.3f} Å")
    else:
        print("⚠️ 没有计算出 RMSD，有效样本为 0")



def main(opt):

    output_dir = get_output_dir(opt.model_dir, opt.experiment_name)
    copy_source(__file__, output_dir)

    opt.dist_url = f'tcp://{opt.node}:{opt.port}'
    print('Using url {}'.format(opt.dist_url))

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(test, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        test(opt.gpu, opt, output_dir)


def test(gpu, opt, output_dir):
    logger = setup_logging(output_dir)

    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method=opt.dist_url,
            world_size=opt.world_size,
            rank=opt.rank
        )

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

    ''' create networks '''
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == 'multi':
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)
    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)
        logger.info("Model = %s" % str(model))
        total_params = sum(p.numel() for p in model.parameters())/1e6
        logger.info("Total_params = %.2f M" % total_params)

    model.eval()

    evaluator = Evaluator(results_dir=output_dir)

    with torch.no_grad():
        if should_diag:
            logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model, map_location=f"cuda:{gpu}")
        model.load_state_dict(resumed_param['model_state'])

        opt.eval_path = os.path.join(outf_syn, 'samples_19999.pth')
        Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)

        if opt.eval_mode == "forward_rmsd":
            forward_rmsd_eval(model, opt, gpu, output_dir)
        elif opt.eval_mode == "forward_denoise":
            stats = forward_denoise_eval(model, opt, gpu, outf_syn, evaluator)
        elif opt.eval_mode == "sample":
            stats = generate_peptide_eval(model, opt, gpu, outf_syn, evaluator)
        else:
            raise ValueError(f"Unknown eval_mode: {opt.eval_mode}")

        if should_diag:
            logger.info(stats)

        

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='dit3d_raw', help='experiment name (used for checkpointing and logging)')

    parser.add_argument('--dataroot', default='/root/DiT-3D/data/peptides_only_ligand_raw')
    parser.add_argument('--category', default='peptide')
    parser.add_argument('--num_classes', type=int, default=55)

    parser.add_argument('--bs', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument("--voxel_size", type=int, choices=[16, 32, 64], default=32)

    '''model'''
    parser.add_argument("--model_type", type=str, choices=list(DiT3D_models.keys()), default="DiT-S/4")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--model', default='',required=True, help="path to model (to continue training)")

    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    # parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
    #                     help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. None means using all available GPUs.')
    
    '''eval'''

    parser.add_argument('--eval_path',
                        default='')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')
    parser.add_argument('--eval_mode', default='sample', choices=['sample', 'forward_denoise', 'forward_rmsd'],
                    help='Evaluation mode: sample (default) / forward_denoise / forward_rmsd')


    opt = parser.parse_args()


    return opt
if __name__ == '__main__':
    opt = parse_args()
    set_seed(opt)

    main(opt)
