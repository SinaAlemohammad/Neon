#!/usr/bin/env python3
"""
calculate_fid.py — EDM sampler, single checkpoint, distributed, print-only FID (URL-aware ref).

Usage (example):
  torchrun --standalone --nproc_per_node=8 edm/calculate_fid.py \
    --network_pkl edm/Neon_EDM_Conditional_CIFAR10.pkl \
    --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz \
    --seeds 0-49999 --max_batch_size 256 --num_steps 18

Notes
- Generates images in memory, computes FID/IS-like stats via Inception features, prints ONE line, saves nothing.
- Works with local paths or URLs for both --network_pkl and --ref (uses dnnlib.util.open_url).
"""

import os
import re
import pickle
import warnings
import numpy as np
import scipy.linalg
import click
import torch
import dnnlib
import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# EDM sampler (same update rule as in your original script)
@torch.no_grad()
def edm_sampler(net, latents, class_labels=None, randn_like=torch.randn_like,
                num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

# -----------------------------------------------------------------------------
# Helpers
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(s) % (1 << 32)) for s in seeds]
    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=g, **kwargs) for g in self.generators])
    def randn_like(self, x):
        return self.randn(x.shape, dtype=x.dtype, layout=x.layout, device=x.device)
    def randint(self, high, *, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(high, size=size[1:], generator=g, **kwargs) for g in self.generators])

def parse_int_list(s):
    if isinstance(s, list):
        return s
    out, rgx = [], re.compile(r'^(\d+)-(\d+)$')
    for p in str(s).split(','):
        p = p.strip()
        m = rgx.match(p)
        if m: out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else: out.append(int(p))
    return out

class GeneratedImagesDataset(Dataset):
    def __init__(self, images): self.images = images
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 3 and img.shape[0] != 3:
            img = np.transpose(img, (2, 0, 1))
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        return torch.from_numpy(img).to(torch.uint8), 0

def calculate_inception_stats_from_dataset(dataset, detector_net, batch_size=64, num_workers=0, device=torch.device('cuda')):
    detector_kwargs = dict(return_features=True)
    feat_dim = 2048
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers)

    sum_feats = torch.zeros([feat_dim], dtype=torch.float64, device=device)
    sum_outer = torch.zeros([feat_dim, feat_dim], dtype=torch.float64, device=device)
    count = 0

    for images, _ in tqdm.tqdm(loader, unit='batch', disable=True):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        feats = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        sum_feats += feats.sum(0)
        sum_outer += feats.T @ feats
        count += feats.shape[0]

    if dist.is_initialized():
        t_count = torch.tensor(count, dtype=torch.float64, device=device)
        dist.all_reduce(t_count); count = int(t_count.item())
        dist.all_reduce(sum_feats)
        dist.all_reduce(sum_outer)

    mu = sum_feats / count
    cov = (sum_outer / count) - torch.outer(mu, mu)
    cov = cov * (count / max(count - 1, 1))  # unbiased covariance
    return mu.cpu().numpy(), cov.cpu().numpy(), count

def fid_from_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(sigma @ sigma_ref, disp=False)
    if np.iscomplexobj(s):
        s = s.real
    return float(m + np.trace(sigma + sigma_ref - 2 * s))

# -----------------------------------------------------------------------------
# CLI
@click.command()
@click.option('--network_pkl', help='Network pickle (PATH or URL)', required=True)
@click.option('--ref',         help='Reference NPZ with keys {mu,sigma} (PATH or URL)', required=True)
@click.option('--seeds',       help='Seeds, e.g. "0-49999" or "1,2,5-10"', default='0-9999')
@click.option('--max_batch_size', type=int, default=64)
@click.option('--class', 'class_idx', type=int, default=None, help='Fixed class index (default: random)')
@click.option('--num_steps',   type=int, default=18)
@click.option('--sigma_min',   type=float, default=0.002)
@click.option('--sigma_max',   type=float, default=80.0)
@click.option('--rho',         type=float, default=7.0)
@click.option('--s_churn',     type=float, default=0.0)
@click.option('--s_min',       type=float, default=0.0)
@click.option('--s_max',       type=float, default=float('inf'))
@click.option('--s_noise',     type=float, default=1.0)
def main(network_pkl, ref, seeds, max_batch_size, class_idx,
         num_steps, sigma_min, sigma_max, rho, s_churn, s_min, s_max, s_noise):

    # ---- distributed init ----
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank(); world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # ---- load model (URL or path) ----
    with dnnlib.util.open_url(network_pkl, verbose=(rank == 0)) as f:
        data = pickle.load(f)
    net = data['ema'].eval().to(device)

    # ---- load inception (fixed URL from NVIDIA) ----
    detector_url = ('https://api.ngc.nvidia.com/v2/models/nvidia/research/'
                    'stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl')
    with dnnlib.util.open_url(detector_url, verbose=(rank == 0)) as f:
        detector = pickle.load(f).eval().to(device)

    # ---- sampler args ----
    sampler_kwargs = dict(
        num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
        S_churn=s_churn, S_min=s_min, S_max=s_max, S_noise=s_noise
    )

    # ---- generate locally ----
    seeds_list = parse_int_list(seeds)
    local_seeds = seeds_list[rank::world]
    images_local = []
    pbar = tqdm.tqdm(range(0, len(local_seeds), max_batch_size),
                     desc=f"rank{rank}", disable=(rank != 0))
    for i in pbar:
        bs = min(max_batch_size, len(local_seeds) - i)
        batch_seeds = local_seeds[i:i+bs]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([bs, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        if net.label_dim:
            if class_idx is None:
                labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[bs], device=device)]
            else:
                labels = torch.zeros([bs, net.label_dim], device=device); labels[:, class_idx] = 1
        else:
            labels = None
        imgs = edm_sampler(net, latents, labels, randn_like=rnd.randn_like, **sampler_kwargs)
        imgs_u8 = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        images_local.extend(imgs_u8)

    # ---- gather to rank0 ----
    all_imgs = [None] * world
    dist.all_gather_object(all_imgs, images_local)
    if rank == 0:
        merged = [im for sub in all_imgs for im in sub]
    else:
        merged = None
    obj = [merged]; dist.broadcast_object_list(obj, src=0); merged = obj[0]

    # ---- FID (ref can be URL or path) ----
    dataset = GeneratedImagesDataset(merged)
    mu, sigma, n_imgs = calculate_inception_stats_from_dataset(
        dataset, detector, batch_size=max_batch_size, device=device
    )

    if rank == 0:
        # URL-aware loading of NPZ reference
        with dnnlib.util.open_url(ref, verbose=False) as f:
            ref_stats = dict(np.load(f))
        fid = fid_from_stats(mu, sigma, ref_stats['mu'], ref_stats['sigma'])
        model_type = "conditional" if net.label_dim else "unconditional"
        print(f"FID={fid:.6f} | {model_type} | steps={num_steps}")

    # ---- teardown ----
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()