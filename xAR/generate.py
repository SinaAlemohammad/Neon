#!/usr/bin/env python
# generate_images.py
#
# Utility that loads a trained xAR model + VAE,
# generates N images, and saves them to an output folder.
# Diagnostic prints and original-grouping label generation.

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from util import misc
from models.vae import AutoencoderKL
from models import xar


def parse_args():
    p = argparse.ArgumentParser('Generate xAR samples', add_help=False)
    p.add_argument('--model', type=str, default='xar_base',
                   choices=[k for k in xar.__dict__.keys() if not k.startswith('_')],
                   help='Which xAR variant to load (must match your checkpoint)')
    p.add_argument('--model_ckpt', type=str, required=True,
                   help='Path to *.pth weights of the xAR model')
    p.add_argument('--vae_path',   type=str, required=True,
                   help='KL‑VAE checkpoint (KL‑16 for ImageNet‑256)')
    p.add_argument('--out_dir',    type=str, required=True,
                   help='Folder to save PNGs')
    p.add_argument('--num_images', type=int, default=50000,
                   help='Total number of images to generate')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Number of images per device per forward pass')
    p.add_argument('--flow_steps', type=int, default=64,
                   help='Number of autoregressive sampling steps')
    p.add_argument('--img_size',   type=int, default=256,
                   help='Image resolution for generation')
    p.add_argument('--cfg',        type=float, default=1.0,
                   help='Classifier‑free guidance scale')
    p.add_argument('--class_num',  type=int, default=1000,
                   help='Number of classes (ImageNet‑1k)')
    p.add_argument('--device',     type=str, default='cuda',
                   help='Device for inference')
    # compatibility with util.misc
    p.add_argument('--dist_on_itp', action='store_true')
    p.add_argument('--dist_url',    default='env://')
    p.add_argument('--local_rank',  default=-1, type=int)
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    misc.init_distributed_mode(args)
    rank = misc.get_rank()
    world = misc.get_world_size()

    # Diagnostic: print CWD and output directory
    cwd = Path('.').absolute()
    out_path = Path(args.out_dir).absolute()
    print(f"[rank {rank}] CWD = {cwd}")
    print(f"[rank {rank}] out_dir = {out_path}")

    # Create/check output directory on rank 0
    if rank == 0:
        out_path.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()

    # Load VAE
    vae = AutoencoderKL(
        embed_dim=16, ch_mult=(1,1,2,2,4),
        ckpt_path=args.vae_path
    ).eval().to(args.device)
    for p in vae.parameters(): p.requires_grad = False

    # Load xAR model variant
    xar_fn = xar.__dict__[args.model]
    model = xar_fn(
        img_size=args.img_size,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        class_num=args.class_num,
        attn_dropout=0.1,
        proj_dropout=0.1,
    ).to(args.device)

    # Load checkpoint
    ckpt = torch.load(args.model_ckpt, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Prepare labels using original grouping logic
    counts = args.num_images // args.class_num
    labels_np = np.arange(args.class_num).repeat(counts)
    
    rem = args.num_images % args.class_num
    if rem:
        labels_np = np.hstack([labels_np, np.zeros(rem, dtype=int)])
    labels = torch.tensor(labels_np, dtype=torch.long)

    # Pad labels to full batches if necessary
    batch_workers = args.batch_size * world
    total = labels.numel()
    if total % batch_workers:
        pad = batch_workers - (total % batch_workers)
        labels = torch.cat([labels, torch.zeros(pad, dtype=torch.long)], 0)
    total_batches = labels.numel() // batch_workers

    start = time.time()
    for b in range(total_batches):
        i0 = b * batch_workers + rank * args.batch_size
        i1 = i0 + args.batch_size
        if i0 >= args.num_images:
            break
        lbl = labels[i0:i1].to(args.device)

        with torch.cuda.amp.autocast():
            sampler = (model.module.sample_tokens if world>1 else model.sample_tokens)
            tokens = sampler(num_steps=args.flow_steps, cfg=args.cfg, label=lbl)
            imgs = vae.decode(tokens / 0.2325)
            imgs = (imgs + 1) / 2

        arr = imgs.mul(255).clamp(0,255).byte().cpu().permute(0,2,3,1).numpy()
        for j, im in enumerate(arr):
            gid = i0 + j
            if gid >= args.num_images:
                break
            save_path = out_path / f"{gid:05d}.png"
            ok = cv2.imwrite(str(save_path), im[:,:,[2,1,0]])
            if not ok:
                print(f"[rank {rank}] ⚠️ cv2.imwrite returned False for {save_path}")
            if not save_path.exists():
                print(f"[rank {rank}] ❌ File not found after write: {save_path}")

        if rank == 0 and (b+1) % 10 == 0:
            done = min((b+1)*args.batch_size*world, args.num_images)
            print(f"[{done}/{args.num_images}] images generated")

    torch.distributed.barrier()
    if rank == 0:
        after = sorted(p.name for p in out_path.glob("*.png"))
        print(f"Took {(time.time() - start)/60:.1f} minutes.")


if __name__ == '__main__':
    main()
