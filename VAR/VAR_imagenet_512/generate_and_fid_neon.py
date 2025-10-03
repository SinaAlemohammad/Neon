#!/usr/bin/env python3
# generate_and_fid_sims.py
#


import os
import time
# Enable synchronous CUDA launches for easier debugging (no printouts here)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import warnings
# suppress that CuDNN workaround warning
warnings.filterwarnings(
    "ignore",
    message="Applied workaround for CuDNN issue.*",
    category=UserWarning,
)

import argparse
import math
import random
import shutil
import re
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from PIL import Image
import torch_fidelity
from tqdm import tqdm

from models import build_vae_var      # <- your builder

# -----------------------------------------------------------------------------#
# Helpers                                                                      #
# -----------------------------------------------------------------------------#

def init_distributed():
    """
    Initialize torch.distributed using torchrun-managed rendezvous.
    Avoid explicit port binding to prevent "address already in use" errors.
    """
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
        rank = local_rank = 0
        world = 1
    return rank, local_rank, world


def save_tensor_as_png(t: torch.Tensor, path: Path):
    img = (t.clamp(0, 1) * 255).to(torch.uint8).cpu()
    arr = img.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(arr).save(path, compress_level=3)


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

# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser("Generate VAR samples, merge ckpts, and compute FID")
    p.add_argument("--N",              type=int,   required=True)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--model_depth",    type=int,   default=36, choices=[36])
    p.add_argument("--cfg",            type=float, default=4.0)
    p.add_argument("--top_k",          type=int,   default=900)
    p.add_argument("--top_p",          type=float, default=0.95)
    p.add_argument("--more_smooth",    action="store_true")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--fid_stats",      type=str,   required=True)
    p.add_argument("--temp_dir",       type=str,   default="temp_var_imgs")
    p.add_argument("--cuda",           action="store_true")
    p.add_argument("--var_ckpt",       type=str,   default="var_d16.pth",
                   help="path to the primary VAR checkpoint (.pth)")
    p.add_argument("--var_aux",        type=str,   required=True,
                   help="path to the auxiliary checkpoint (.pth)")
    p.add_argument("--w",              type=float, required=True,
                   help="merge weight: (1+w)*θ_B - w*θ_A")
    p.add_argument("--out_dir",        type=str,   required=True,
                   help="where to save FID results")
    return p.parse_args()

# -----------------------------------------------------------------------------#
# Main                                                                         #
# -----------------------------------------------------------------------------#
@torch.inference_mode()
def main():
    args = parse_args()

    rank, local_rank, world = init_distributed()
    master = (rank == 0)
    set_seed(args.seed + rank)

    # ----- download ckpts (VAE + primary VAR) -----
    hf = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt = "vae_ch160v4096z32.pth"
    ckpt_list = [vae_ckpt, args.var_ckpt]
    if master:
        for ck in ckpt_list:
            if not os.path.exists(ck):
                os.system(f"wget {hf}/{ck}")
    if world > 1:
        dist.barrier()

    # disable DDP re-init
    setattr(torch.nn.Linear, "reset_parameters", lambda *_: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda *_: None)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ----- build models -----
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=(1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
        num_classes=1000, depth=args.model_depth, shared_aln=True,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
    vae.eval().requires_grad_(False)

    # load θ_B
    raw_B = torch.load(args.var_ckpt, map_location="cpu")
    trainer_sd_B = raw_B.get('trainer', raw_B)
    var_sd_B = trainer_sd_B.get('var_wo_ddp', trainer_sd_B)
    # load θ_A
    raw_A = torch.load(args.var_aux, map_location="cpu")
    trainer_sd_A = raw_A.get('trainer', raw_A)
    var_sd_A = trainer_sd_A.get('var_wo_ddp', trainer_sd_A)

    # stable merge: vB + w*(vB - vA)
    w = args.w
    merged_sd = {}
    for k, vB in var_sd_B.items():
        vA = var_sd_A.get(k)
        if torch.isinf(vB).any() or vA is None:
            merged_sd[k] = vB
        else:
            merged_sd[k] = vB + w * (vB - vA)
    var.load_state_dict(merged_sd, strict=True)
    var.eval().requires_grad_(False)

    # ----- distribute work -----
    base = args.N // world
    rem  = args.N % world
    local_N = base + (1 if rank < rem else 0)
    if not local_N:
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    start_idx = rank * base + min(rank, rem)
    labels = build_balanced_labels(args.N, 1000, args.seed)
    labels_local = labels[start_idx:start_idx+local_N].to(device)

    # ----- temp dir -----
    root = Path(args.temp_dir)
    if master:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
    if world > 1:
        dist.barrier()

    # ----- sampling with pbar -----
    steps = math.ceil(local_N / args.batch_size)
    prog = tqdm(range(steps), desc="Sampling", dynamic_ncols=True, leave=False)
    with torch.autocast("cuda", torch.float16, enabled=True):
        ptr = 0
        for _ in prog:
            cur_B = min(args.batch_size, local_N - ptr)
            if cur_B == 0:
                break
            lbl = labels_local[ptr:ptr+cur_B]
            imgs = var.autoregressive_infer_cfg(
                B=cur_B, label_B=lbl,
                cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
                g_seed=args.seed + ptr, more_smooth=args.more_smooth,
            )
            for i in range(cur_B):
                gid = start_idx + ptr + i
                save_tensor_as_png(imgs[i], root / f"{gid:08d}.png")
            ptr += cur_B
            torch.cuda.synchronize()  # catch errors early
            if master:
                prog.update(1)
    if world > 1:
        dist.barrier()

    # ----- FID computation & save -----
    if master:
        m = torch_fidelity.calculate_metrics(
            input1=str(root), input2=None,
            fid_statistics_file=args.fid_stats,
            cuda=args.cuda, isc=True, fid=True,
            kid=False, prc=False,
            verbose=False
        )
        fid_val = m['frechet_inception_distance']
        aux_num = re.search(r'images_(\d+)', args.var_aux)
        aux_num = aux_num.group(1) if aux_num else 'aux'
        out_sub = Path(args.out_dir) / aux_num
        out_sub.mkdir(parents=True, exist_ok=True)
        (out_sub / f"fid_w{args.w}_cfg{args.cfg}.txt").write_text(f"{fid_val:.4f}\n")
    if world > 1:
        dist.barrier()
    if master:
        shutil.rmtree(root)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
