#!/usr/bin/env python
# generate_var_dataset.py
#
# Utility that loads the VAR model + VAE,
# generates N images *per class*, and saves them into subfolders by class name under a 'train' directory,
# displaying a class-level progress bar.

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from models import build_vae_var

# -----------------------------------------------------------------------------#
# Helpers                                                                      #
# -----------------------------------------------------------------------------#

def init_distributed():
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = local_rank = 0
        world = 1
    return rank, local_rank, world


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_tensor_as_png(t: torch.Tensor, path: Path):
    img = (t.clamp(0, 1) * 255).to(torch.uint8).cpu()
    arr = img.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(arr).save(path, compress_level=3)

# -----------------------------------------------------------------------------#
# Argument parsing                                                             #
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser('Generate VAR samples per class')
    p.add_argument('--out_dir',     type=str, required=True,
                   help='Root folder to save PNGs (will create train/class subfolders)')
    p.add_argument('--num_images',  type=int, default=750,
                   help='Number of images to generate per class')
    p.add_argument('--batch_size',  type=int, default=64,
                   help='Batch size per device')
    p.add_argument('--cfg',         type=float, default=4.0,
                   help='Classifier-free guidance scale')
    p.add_argument('--top_k',       type=int, default=900,
                   help='Top-k sampling')
    p.add_argument('--top_p',       type=float, default=0.95,
                   help='Top-p sampling')
    p.add_argument('--more_smooth', action='store_true',
                   help='Use the more_smooth flag in VAR sampler')
    p.add_argument('--model_depth', type=int, default=16, choices=[16],
                   help='VAR model depth (affects checkpoint name)')
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--temp_dir',    type=str, default='temp_var_imgs',
                   help='Temporary directory (will be removed)')
    p.add_argument('--cuda',        action='store_true',
                   help='Run FID on GPU (not used here)')
    return p.parse_args()

# -----------------------------------------------------------------------------#
# Main                                                                          #
# -----------------------------------------------------------------------------#
@torch.inference_mode()
def main():
    args = parse_args()
    rank, local_rank, world = init_distributed()
    master = (rank == 0)
    set_seed(args.seed + rank)

    # constants
    VAE_CKPT   = 'vae_ch160v4096z32.pth'
    VAR_CKPT   = f'var_d{args.model_depth}.pth'
    CLASS_MAP  = 'imagenet/imagenet_class_to_idx.json'

    # load class mapping
    class_map = json.load(open(CLASS_MAP, 'r'))
    class_list = sorted(class_map, key=lambda k: class_map[k])

    # prepare output folders under 'train'
    out_root = Path(args.out_dir) / 'train'
    if master:
        shutil.rmtree(out_root, ignore_errors=True)
        for cls in class_list:
            (out_root / cls).mkdir(parents=True, exist_ok=True)
        pbar = tqdm(total=len(class_list), desc='Classes', unit='cls')
    dist.barrier()

    # download checkpoints if needed (only on master)
    hf = "https://huggingface.co/FoundationVision/var/resolve/main"
    if master:
        for ck in (VAE_CKPT, VAR_CKPT):
            if not os.path.exists(ck):
                os.system(f"wget {hf}/{ck}")
    dist.barrier()

    # build models
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setattr(torch.nn.Linear, "reset_parameters", lambda *_: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda *_: None)

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=(1,2,3,4,5,6,8,10,13,16),
        num_classes=len(class_list), depth=args.model_depth, shared_aln=False,
    )

    ckpt = torch.load(VAR_CKPT, map_location='cpu')


    sd = ckpt.get('var_wo_ddp', ckpt.get('state_dict', ckpt))


    var.load_state_dict(sd, strict=True)

    
    vae.load_state_dict(torch.load(VAE_CKPT, map_location="cpu"))
    #var.load_state_dict(torch.load(VAR_CKPT, map_location="cpu"))
    vae.eval().requires_grad_(False)
    var.eval().requires_grad_(False)

    # per-class generation
    for cls_idx, cls_name in enumerate(class_list):
        N = args.num_images
        base = N // world
        rem  = N % world
        local_N = base + (1 if rank < rem else 0)
        start = rank * base + min(rank, rem)
        if local_N == 0:
            dist.barrier()
            if master: pbar.update(1)
            continue

        labels = torch.full((local_N,), cls_idx, device=device, dtype=torch.long)
        steps = (local_N + args.batch_size - 1) // args.batch_size
        ptr = 0
        for _ in range(steps):
            cur_B = min(args.batch_size, local_N - ptr)
            lbl = labels[ptr:ptr+cur_B]
            imgs = var.autoregressive_infer_cfg(
                B=cur_B, label_B=lbl,
                cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
                g_seed=args.seed + start + ptr, more_smooth=args.more_smooth,
            )
            for i in range(cur_B):
                gid = start + ptr + i
                save_tensor_as_png(imgs[i], out_root / cls_name / f"{gid:05d}.png")
            ptr += cur_B

        dist.barrier()
        if master:
            pbar.update(1)

    if master:
        pbar.close()
        print(f"Generated {args.num_images} images for each of {len(class_list)} classes under '{args.out_dir}/train'.")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
