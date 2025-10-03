#!/usr/bin/env python3
# calculate_fid.py

import os, shutil, pickle, functools, warnings, contextlib, subprocess, urllib.request, tempfile
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torch_fidelity
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm

import dnnlib
from torch_utils import misc

warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")
warnings.filterwarnings("ignore", category=FutureWarning)

# Quiet worker logs (torchrun banner may print before Python starts)
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("NCCL_DEBUG", "ERROR")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

DEFAULT_FID_STATS = "fid_stats/adm_in256_stats.npz"

# ----------------------------------------------------------------------------
# Generators (IMM-compatible)

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
        nt_max = net.get_log_nt(torch.as_tensor(net.T,   dtype=torch.float32, device=dev)).exp().item()
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

# -------------------- neon merge helper (inf-safe) --------------------
def build_neon_state_dict(base_module: torch.nn.Module, aux_module: torch.nn.Module, w: float, device: torch.device):
    b_sd = base_module.state_dict()
    a_sd = aux_module.state_dict()
    merged = {}
    for k, b in b_sd.items():
        if not isinstance(b, torch.Tensor):
            merged[k] = b
            continue
        b_t = b.to(device=device)
        if k in a_sd and isinstance(a_sd[k], torch.Tensor) and torch.is_floating_point(b_t):
            a_t = a_sd[k].to(device=device, dtype=b_t.dtype)
            m = b_t.clone()
            finite_mask = torch.isfinite(b_t) & torch.isfinite(a_t)
            if finite_mask.any():
                # (1+w)*θ_B - w*θ_A == θ_B - w*(θ_A - θ_B)
                m[finite_mask] = b_t[finite_mask] - w * (a_t[finite_mask] - b_t[finite_mask])
            merged[k] = m
        else:
            merged[k] = b_t
    return merged

# ----------------------------------------------------------------------------

def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Try Python
    try:
        urllib.request.urlretrieve(url, dest)
        return
    except Exception:
        pass
    # Fallback to wget/curl
    for cmd in (["wget", "-q", "-O", str(dest), url],
                ["curl", "-L", "-s", "-o", str(dest), url]):
        try:
            subprocess.check_call(cmd)
            return
        except Exception:
            continue
    raise RuntimeError(f"Failed to download {url} to {dest}")

@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    # -------------------- DDP setup --------------------
    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_cuda else 0
    if use_cuda:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
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

    preset_key     = str(getattr(config, "preset", "8_steps_cfg1.5_pushforward_uniform"))
    fid_stats_path = str(getattr(config, "fid_stats", DEFAULT_FID_STATS))

    per_class_count = int(getattr(config, "per_class_count", 0))
    if per_class_count <= 0 and int(config.label_dim) > 0:
        if rank == 0:
            raise ValueError("Provide +per_class_count=<int> on the CLI for class-conditional generation.")

    # -------------------- RNG / Backend --------------------
    seed = config.eval.seed if config.eval.seed is not None else 42
    seed = seed + rank
    np.random.seed(seed % (1 << 31))
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = bool(config.eval.cudnn_benchmark)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # -------------------- Encoder --------------------
    encoder = dnnlib.util.construct_class_by_name(**config.encoder)

    # -------------------- Network --------------------
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

    aux_resume = getattr(config, "aux_resume", None)
    neon_w = float(getattr(config, "neon_w", getattr(config.eval, "neon_w", 0.0) or 0.0))

    # Load base (EMA module expected)
    with dnnlib.util.open_url(resume_pkl, verbose=(rank == 0)) as f:
        base_data = pickle.load(f)
    base_module = base_data.get('ema', None)
    if base_module is None:
        raise KeyError("Base checkpoint missing 'ema' module.")

    # If no wrapper, use EMA directly
    if net is None:
        net = base_module.eval().requires_grad_(False).to(device)

    # Neon merge if requested (both aux and nonzero weight)
    if aux_resume is not None and neon_w != 0.0:
        with dnnlib.util.open_url(aux_resume, verbose=(rank == 0)) as f:
            aux_data = pickle.load(f)
        aux_module = aux_data.get('ema', None)
        if aux_module is None:
            raise KeyError("Aux checkpoint missing 'ema' module.")
        merged_sd = build_neon_state_dict(base_module, aux_module, neon_w, device)
        missing, unexpected = net.load_state_dict(merged_sd, strict=False)
        if (missing or unexpected) and rank == 0:
            warnings.warn(
                f"load_state_dict: missing={missing[:5]}(…{len(missing)}), "
                f"unexpected={unexpected[:5]}(…{len(unexpected)})"
            )
    else:
        misc.copy_params_and_buffers(src_module=base_module, dst_module=net, require_all=True)

    # -------------------- Sampling preset (dynamic) --------------------
    sample_block = config.get('sampling', {}).get(preset_key, None)
    if sample_block is None:
        available = list(config.get('sampling', {}).keys())
        if rank == 0:
            raise KeyError(f"Sampling preset '{preset_key}' not found. Available: {available}")

    cfg_override = getattr(config, "cfg_scale", None)
    if cfg_override is not None:
        sample_block = dict(sample_block)
        sample_block["cfg_scale"] = float(cfg_override)

    gen_fn = functools.partial(generator_fn, **sample_block)

    # -------------------- Shapes & classes --------------------
    bs = int(config.eval.batch_size)
    H = net.img_resolution
    W = net.img_resolution
    C = net.img_channels
    num_classes = int(net.label_dim)

    # -------------------- Global label sequence & unique IDs --------------------
    if num_classes > 0:
        labels_all = np.repeat(np.arange(num_classes, dtype=np.int64), per_class_count)
        unique_ids = np.concatenate([np.arange(per_class_count, dtype=np.int64) for _ in range(num_classes)])
        idx_all = np.arange(labels_all.shape[0], dtype=np.int64)
        local_idx = idx_all[rank::world_size]
        local_labels = labels_all[local_idx]
        local_unique = unique_ids[local_idx]
        total_local = local_labels.shape[0]
    else:
        total_images = int(getattr(config, "total_images", 0))
        if total_images <= 0:
            if rank == 0:
                raise ValueError("For unconditional models set +total_images=<int> (since label_dim==0).")
        idx_all = np.arange(total_images, dtype=np.int64)
        local_idx = idx_all[rank::world_size]
        local_labels = np.array([], dtype=np.int64)  # no labels
        local_unique = local_idx
        total_local = local_idx.shape[0]

    # -------------------- Rank-0 temp dir broadcast (no persistent saves) ----
    if rank == 0:
        out_dir = tempfile.mkdtemp(prefix="imm_fid_")
    else:
        out_dir = None
    if dist.is_initialized():
        obj_list = [out_dir]
        dist.broadcast_object_list(obj_list, src=0)
        out_dir = obj_list[0]

    # -------------------- Progress bar --------------------
    total_batches = ceil(total_local / bs)
    pbar = tqdm(total=total_batches,
                desc=f"Rank {rank}",
                disable=(world_size > 1 and rank != 0),
                leave=True,
                dynamic_ncols=True)

    # -------------------- Helpers --------------------
    one_hot_buf = torch.empty(bs, num_classes, device=device, dtype=torch.float32) if num_classes > 0 else None

    def make_one_hot_np(labels_np: np.ndarray, out: torch.Tensor):
        out.zero_()
        idx = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
        out[:idx.numel()].scatter_(1, idx.view(-1, 1), 1.0)
        return out[:idx.numel()]

    io_pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 8))

    def save_batch_cpu(imgs_u8_cpu: torch.Tensor, labels_np: np.ndarray, uniq_np: np.ndarray, out_dir: str):
        n = imgs_u8_cpu.size(0)
        for i in range(n):
            if num_classes > 0:
                cls = int(labels_np[i]); uniq = int(uniq_np[i])
                fname = f"label{cls}_{uniq:06d}.png"
            else:
                uniq = int(uniq_np[i]); fname = f"sample_{uniq:06d}.png"
            path = os.path.join(out_dir, fname)
            arr = imgs_u8_cpu[i].permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(path, compress_level=1, optimize=False)

    prev_future = None

    # -------------------- Batch loop (no waste, autocast FP16) --------------------
    for start in range(0, total_local, bs):
        end = min(start + bs, total_local)
        cur_unique = local_unique[start:end]
        cur_bs = cur_unique.shape[0]

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            z = net.get_init_noise([cur_bs, C, H, W], device)
            if num_classes > 0:
                cur_labels = local_labels[start:end]
                c = make_one_hot_np(cur_labels, one_hot_buf)
            else:
                cur_labels = np.array([], dtype=np.int64)
                c = None

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
        prev_future = io_pool.submit(
            save_batch_cpu, imgs_cpu,
            cur_labels.copy() if num_classes > 0 else np.empty((cur_bs,), dtype=np.int64),
            cur_unique.copy(), out_dir
        )
        pbar.update(1)

    if prev_future is not None:
        prev_future.result()
    io_pool.shutdown(wait=True)
    pbar.close()

    # -------------------- FID (rank-0), print-only, then cleanup ---------------
    if world_size > 1:
        ddp_barrier()
    if rank == 0:
        try:
            img_count = sum(1 for p in os.scandir(out_dir)
                            if p.is_file() and p.name.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')))
        except FileNotFoundError:
            img_count = 0

        metrics = torch_fidelity.calculate_metrics(
            input1=out_dir,
            input2=None,
            fid_statistics_file=fid_stats_path,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            verbose=False
        )
        fid_val = float(metrics['frechet_inception_distance'])
        is_mean = float(metrics.get('inception_score_mean', float('nan')))
        is_std  = float(metrics.get('inception_score_std', float('nan')))


        print(f"FID={fid_val:.6f} | IS={is_mean:.6f}±{is_std:.6f} | images={img_count} | preset={preset_key}")

        # Always clean images (remove the temp directory)
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: failed to remove temp dir {out_dir}: {e}")

    if world_size > 1:
        ddp_barrier()

    # ---- clean DDP teardown ----
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        with contextlib.suppress(Exception):
            dist.destroy_process_group()

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
