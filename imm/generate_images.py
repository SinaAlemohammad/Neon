# generate_images.py — fast DDP sampler (8-step), no waste, BF16/TF32, overlapped I/O,
# explicit NCCL device_ids on barrier to silence warnings.
import os
import pickle
import functools
import warnings
from math import ceil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm

import dnnlib
from torch_utils import misc

warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")

# ----------------------------------------------------------------------------
# IMM-compatible sampling (dtype-friendly for autocast)

def generator_fn(*args, name='pushforward_generator_fn', **kwargs):
    return globals()[name](*args, **kwargs)

@torch.no_grad()
def pushforward_generator_fn(net, latents, class_labels=None, discretization=None, mid_nt=None, num_steps=None, cfg_scale=None):
    d = latents.dtype
    dev = latents.device

    if discretization == 'uniform':
        t_steps = torch.linspace(net.T, net.eps, num_steps + 1, dtype=d, device=dev)
    elif discretization == 'edm':
        nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float32, device=dev)).exp().item()
        nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float32, device=dev)).exp().item()
        rho = 7.0
        step_indices = torch.arange(num_steps + 1, dtype=torch.float32, device=dev)
        nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
        t_steps = net.nt_to_t(nt_steps).to(d)
    else:
        if mid_nt is None:
            mid_nt = []
        mid_t = [net.nt_to_t(torch.as_tensor(nt, dtype=torch.float32, device=dev)).item() for nt in mid_nt]
        t_steps = torch.tensor([net.T] + list(mid_t), dtype=d, device=dev)
        t_steps = torch.cat([t_steps, torch.ones_like(t_steps[:1]) * net.eps])

    x = latents
    for (t_cur, t_next) in zip(t_steps[:-1], t_steps[1:]):
        x = net.cfg_forward(x, t_cur, t_next, class_labels=class_labels, cfg_scale=cfg_scale)
    return x

# ----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    # -------------------- DDP setup --------------------
    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_cuda else 0
    if use_cuda:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        backend = dist.get_backend()
    else:
        rank, world_size = 0, 1
        device = torch.device("cpu")
        backend = None

    def ddp_barrier():
        if dist.is_initialized():
            if backend == "nccl":
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()

    # -------------------- Config & CLI overrides --------------------
    config = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    per_class_count = int(getattr(config, "per_class_count", 0))
    out_dir = str(getattr(config, "out_dir", ""))

    if per_class_count <= 0 or not out_dir:
        if rank == 0:
            raise ValueError("Provide +per_class_count=<int> and +out_dir=<path> on the CLI.")

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if world_size > 1:
        ddp_barrier()

    # -------------------- RNG / Backend --------------------
    seed = config.eval.seed
    if seed is None:
        seed = 42
    seed = seed + rank
    np.random.seed(seed % (1 << 31))
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = bool(config.eval.cudnn_benchmark)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # -------------------- Encoder --------------------
    # if rank == 0:
    #     print('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**config.encoder)

    # -------------------- Network --------------------
    # if rank == 0:
    #     print("Constructing network...")

    interface_kwargs = dict(
        img_resolution=config.resolution,
        img_channels=config.channels,
        label_dim=config.label_dim,
    )

    if config.get('network', None) is not None:
        net = dnnlib.util.construct_class_by_name(**config.network, **interface_kwargs)
        net.eval().requires_grad_(False).to(device)
    else:
        net = None

    resume_pkl = config.eval.resume
    if resume_pkl is None:
        if rank == 0:
            raise ValueError("Set eval.resume to your checkpoint path (e.g., imagenet256_ts_a2.pkl).")

    with dnnlib.util.open_url(resume_pkl, verbose=(rank == 0)) as f:
        data = pickle.load(f)

    if net is not None:
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=True)
    else:
        net = data['ema'].eval().requires_grad_(False).to(device)

    # -------------------- Force the 8-step preset --------------------
    preset_key = "8_steps_cfg1.5_pushforward_uniform"
    sample_block = config.get('sampling', {}).get(preset_key, None)
    if sample_block is None:
        available = list(config.get('sampling', {}).keys())
        if rank == 0:
            raise KeyError(f"Sampling preset '{preset_key}' not found. Available: {available}")
    gen_fn = functools.partial(generator_fn, **sample_block)

    # -------------------- Shapes & classes --------------------
    bs = int(config.eval.batch_size)
    H = net.img_resolution
    W = net.img_resolution
    C = net.img_channels
    num_classes = int(net.label_dim)


    # -------------------- Global label sequence & unique IDs --------------------
    labels_all = np.repeat(np.arange(num_classes, dtype=np.int64), per_class_count)
    unique_ids = np.concatenate([np.arange(per_class_count, dtype=np.int64) for _ in range(num_classes)])

    idx_all = np.arange(labels_all.shape[0], dtype=np.int64)
    local_idx = idx_all[rank::world_size]
    local_labels = labels_all[local_idx]
    local_unique = unique_ids[local_idx]

    total_local = local_labels.shape[0]
    total_batches = ceil(total_local / bs)

    pbar = tqdm(total=total_batches, desc=f"Rank {rank}", disable=(world_size > 1 and rank != 0))

    # -------------------- Helpers --------------------
    one_hot_buf = torch.empty(bs, num_classes, device=device, dtype=torch.float32)

    def make_one_hot_np(labels_np: np.ndarray, out: torch.Tensor):
        out.zero_()
        idx = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
        out[:idx.numel()].scatter_(1, idx.view(-1, 1), 1.0)
        return out[:idx.numel()]

    io_pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 8))

    def save_batch_cpu(imgs_u8_cpu: torch.Tensor, labels_np: np.ndarray, uniq_np: np.ndarray, out_dir: str):
        n = imgs_u8_cpu.size(0)
        for i in range(n):
            cls = int(labels_np[i]); uniq = int(uniq_np[i])
            fname = f"label{cls}_{uniq:06d}.png"
            path = os.path.join(out_dir, fname)
            arr = imgs_u8_cpu[i].permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(path, compress_level=1, optimize=False)

    prev_future = None

    # -------------------- Batch loop (no waste, autocast BF16) --------------------
    for start in range(0, total_local, bs):
        end = min(start + bs, total_local)
        cur_labels = local_labels[start:end]
        cur_unique = local_unique[start:end]
        cur_bs = cur_labels.shape[0]

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            z = net.get_init_noise([cur_bs, C, H, W], device)
            c = make_one_hot_np(cur_labels, one_hot_buf) if num_classes > 0 else None

            lat = gen_fn(net, z, c)
            imgs = encoder.decode(lat).detach()

        if imgs.dtype == torch.uint8:
            imgs_cpu = imgs.cpu()
        else:
            x = imgs.float()
            if x.min() >= 0.0 and x.max() <= 1.0:
                imgs_cpu = (x * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            else:
                imgs_cpu = ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()

        if prev_future is not None:
            prev_future.result()
        prev_future = io_pool.submit(save_batch_cpu, imgs_cpu, cur_labels.copy(), cur_unique.copy(), out_dir)

        pbar.update(1)

    if prev_future is not None:
        prev_future.result()
    io_pool.shutdown(wait=True)
    pbar.close()

    if world_size > 1:
        ddp_barrier()
    if rank == 0:
        # print("Done.")
        pass

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()