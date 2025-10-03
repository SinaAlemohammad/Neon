#!/usr/bin/env python3
# calculate_fid.py

import os
# Quiet worker logs (torchrun banner still prints before Python starts unless OMP is set in shell)
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("NCCL_DEBUG", "ERROR")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import argparse
import math
import random
import shutil
import contextlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import urllib.request
import subprocess
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torch_fidelity
from torch.amp import autocast
from tqdm import tqdm

# Make sure PYTHONPATH points to the repo that contains `models/`
from models import build_vae_var


# -------------------- helpers --------------------
def init_distributed():
    """Initialize torch.distributed using torchrun; select CUDA device early."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world = 1
    return rank, local_rank, world


def ddp_barrier(local_rank: int):
    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()


def save_tensor_as_png(t: torch.Tensor, path: Path):
    img = (t.clamp(0, 1) * 255).to(torch.uint8).cpu()
    Image.fromarray(img.permute(1, 2, 0).contiguous().numpy()).save(path, compress_level=3)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_balanced_labels(num_images: int, num_classes: int, seed: int) -> torch.Tensor:
    per_cls = num_images // num_classes
    labels = torch.arange(num_classes).repeat_interleave(per_cls)
    rem = num_images - labels.numel()
    if rem:
        labels = torch.cat([labels, torch.randint(num_classes, (rem,))])
    g = torch.Generator().manual_seed(seed)
    return labels[torch.randperm(len(labels), generator=g)]


def load_torch(path):
    """torch.load preferring weights_only=True (newer PyTorch), fallback if needed."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Try Python first
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


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser("Generate Neon VAR samples (single model) + FID/IS")
    p.add_argument("--N", type=int, required=True, help="total images to generate")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model_depth", type=int, default=36, choices=[36], help="VAR depth (d36)")
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--top_k", type=int, default=900)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--more_smooth", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--fid_stats", type=str, required=True, help="npz stats path")
    p.add_argument("--temp_dir", type=str, default="temp_var_imgs", help="temporary image dir (deleted)")
    p.add_argument("--var_ckpt", type=str, required=True, help="path to VAR checkpoint (.pth)")

    # If omitted or file missing, we auto-resolve and download the official VAE.
    p.add_argument("--vae_ckpt", type=str, default=None, help="path to VAE checkpoint (optional)")

    # --cuda flag is ignored; we always run metrics on CUDA
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


# -------------------- main --------------------
@torch.inference_mode()
def main():
    args = parse_args()

    # Enforce CUDA metrics
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch_fidelity metrics, but no CUDA device is available.")

    rank, local_rank, world = init_distributed()
    master = (rank == 0)
    set_seed(args.seed + rank)

    script_dir = Path(__file__).resolve().parent
    repo_default = script_dir / "pretrained" / "vae_ch160v4096z32.pth"
    cwd_default = Path("vae_ch160v4096z32.pth")
    if args.vae_ckpt and Path(args.vae_ckpt).exists():
        vae_ckpt_path = Path(args.vae_ckpt)
    elif cwd_default.exists():
        vae_ckpt_path = cwd_default
    elif repo_default.exists():
        vae_ckpt_path = repo_default
    else:
        if master:
            url = "https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth"
            download_file(url, repo_default)
        ddp_barrier(local_rank)
        vae_ckpt_path = repo_default

    # Disable DDP re-init (as in Neon code)
    setattr(torch.nn.Linear, "reset_parameters", lambda *_: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda *_: None)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---- build models (matches your settings) ----
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device,
        patch_nums=(1, 2, 3, 4, 6, 9, 13, 18, 24, 32),  
        num_classes=1000,
        depth=args.model_depth,                         
        shared_aln=True,                               
    )
    # VAE weights
    vae_sd = load_torch(str(vae_ckpt_path))
    vae.load_state_dict(vae_sd["model"] if (isinstance(vae_sd, dict) and "model" in vae_sd) else vae_sd, strict=True)
    vae.eval().requires_grad_(False)

    # VAR weights (support raw or wrapped under trainer/var_wo_ddp)
    raw = load_torch(args.var_ckpt)
    sd = raw.get("trainer", raw) if isinstance(raw, dict) else raw
    sd = sd.get("var_wo_ddp", sd) if isinstance(sd, dict) else sd
    var.load_state_dict(sd, strict=True)
    var.eval().requires_grad_(False)

    # ---- shard work across ranks ----
    total = args.N
    base = total // world
    rem = total % world
    local_N = base + (1 if rank < rem else 0)
    if local_N == 0:
        ddp_barrier(local_rank)
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
        return

    start_idx = rank * base + min(rank, rem)
    labels = build_balanced_labels(total, 1000, args.seed)
    labels_local = labels[start_idx:start_idx + local_N].to(device)

    # ---- temp dir ----
    root = Path(args.temp_dir)
    if master:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
    ddp_barrier(local_rank)

    # ---- sampling ----
    steps = math.ceil(local_N / args.batch_size)
    pbar = tqdm(range(steps), desc=f"Gen-{Path(args.var_ckpt).stem}", dynamic_ncols=True, leave=True) if master else range(steps)

    ptr = 0
    with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        for _ in pbar:
            cur_B = min(args.batch_size, local_N - ptr)
            if cur_B <= 0:
                continue  # keep loop count consistent across ranks
            lbl = labels_local[ptr:ptr + cur_B]
            imgs = var.autoregressive_infer_cfg(
                B=cur_B, label_B=lbl,
                cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
                g_seed=args.seed + ptr, more_smooth=args.more_smooth,
            )
            for i in range(cur_B):
                gid = start_idx + ptr + i
                if gid >= total:
                    break
                save_tensor_as_png(imgs[i], root / f"{gid:08d}.png")
            ptr += cur_B
            torch.cuda.synchronize()

    ddp_barrier(local_rank)

    # ---- metrics (master) ----
    if master:
        m = torch_fidelity.calculate_metrics(
            input1=str(root), input2=None,
            fid_statistics_file=args.fid_stats,
            cuda=True, isc=True, fid=True,
            kid=False, prc=False, verbose=False
        )
        fid_val = float(m["frechet_inception_distance"])
        isc_mean = m.get("inception_score_mean")
        isc_std  = m.get("inception_score_std")

        parts = [f"FID={fid_val:.4f}"]
        if isc_mean is not None and isc_std is not None:
            parts.append(f"IS={isc_mean:.4f}±{isc_std:.4f}")
        parts += [
            f"cfg={args.cfg:g}",
            f"N={args.N}",
            f"depth={args.model_depth}",
            f"ckpt={Path(args.var_ckpt).stem}"
        ]
        print(" | ".join(parts))

    # cleanup images on all ranks after metrics
    ddp_barrier(local_rank)
    if master:
        shutil.rmtree(root, ignore_errors=True)

    # ---- clean DDP teardown ----
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        with contextlib.suppress(Exception):
            ddp_barrier(local_rank)
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
