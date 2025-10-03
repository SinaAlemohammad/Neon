#!/usr/bin/env python
# generate_dataset.py
#
# Utility that loads a trained xAR model + VAE,
# generates N images *per class*, and saves them into subfolders by class name,
# displaying a progress bar per folder.

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from util import misc
from models.vae import AutoencoderKL
from models import xar


def parse_args():
    p = argparse.ArgumentParser('Generate xAR samples per class', add_help=False)
    p.add_argument('--model', type=str, default='xar_base',
                   choices=[k for k in xar.__dict__ if not k.startswith('_')],
                   help='Which xAR variant to load (must match your checkpoint)')
    p.add_argument('--model_ckpt', type=str, required=True,
                   help='Path to *.pth weights of the xAR model')
    p.add_argument('--vae_path', type=str, required=True,
                   help='KL‑VAE checkpoint (KL‑16 for ImageNet‑256)')
    p.add_argument('--out_dir', type=str, required=True,
                   help='Root folder to save PNGs (will create class subfolders)')
    p.add_argument('--num_images', type=int, default=750,
                   help='Number of images to generate per class')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Number of images per device per forward pass')
    p.add_argument('--flow_steps', type=int, default=40,
                   help='Number of autoregressive sampling steps')
    p.add_argument('--img_size', type=int, default=256,
                   help='Image resolution for generation')
    p.add_argument('--cfg', type=float, default=1.0,
                   help='Classifier‑free guidance scale')
    p.add_argument('--class_num', type=int, default=1000,
                   help='Number of classes (ImageNet‑1k)')
    p.add_argument('--class_map', type=str,
                   default='imagenet/imagenet_class_to_idx.json',
                   help='JSON mapping class_name → label index')
    p.add_argument('--device', type=str, default='cuda',
                   help='Device for inference')
    # distributed args (injected by torchrun)
    p.add_argument('--local_rank', type=int, default=-1)
    p.add_argument('--dist_url', default='env://')
    p.add_argument('--dist_on_itp', action='store_true',
                   help='Allow distributed on ITP clusters')
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    misc.init_distributed_mode(args)
    rank = misc.get_rank()
    world = misc.get_world_size()

    # load and invert class map
    class_map = json.load(open(args.class_map, 'r'))
    class_list = sorted(class_map.keys())

    # create output subfolders
    out_root = Path(args.out_dir)
    if rank == 0:
        for cls in class_list:
            (out_root / cls).mkdir(parents=True, exist_ok=True)
        pbar = tqdm(class_list, desc='Classes', unit='cls')
    torch.distributed.barrier()

    # prepare VAE
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1,1,2,2,4), ckpt_path=args.vae_path)
    vae.eval().to(args.device)
    for p in vae.parameters(): p.requires_grad = False

    # prepare xAR model
    xar_fn = xar.__dict__[args.model]
    model = xar_fn(img_size=args.img_size,
                   vae_stride=16, patch_size=1,
                   vae_embed_dim=16, class_num=args.class_num,
                   attn_dropout=0.1, proj_dropout=0.1)
    model = model.to(args.device)
    ckpt = torch.load(args.model_ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    # generate per-class
    for cls_idx, cls_name in enumerate(class_list):
        # per-class desired images
        N = args.num_images
        # label array
        labels_cls = np.full(N, cls_idx, dtype=int)
        # pad to full batches
        batch_w = args.batch_size * world
        pad = (batch_w - (labels_cls.size % batch_w)) % batch_w
        if pad:
            labels_cls = np.hstack([labels_cls, np.zeros(pad, dtype=int)])
        total_batches = labels_cls.size // batch_w

        for b in range(total_batches):
            offset = b * batch_w
            i0 = offset + rank * args.batch_size
            if i0 >= N:
                break
            i1 = i0 + args.batch_size
            lbl = torch.tensor(labels_cls[i0:i1], dtype=torch.long, device=args.device)

            with torch.cuda.amp.autocast():
                sampler = model.module.sample_tokens if world > 1 else model.sample_tokens
                tokens = sampler(num_steps=args.flow_steps, cfg=args.cfg, label=lbl)
                imgs = vae.decode(tokens / 0.2325)
                imgs = (imgs + 1) / 2

            arr = imgs.mul(255).clamp(0,255).byte().cpu().permute(0,2,3,1).numpy()
            for j, im in enumerate(arr):
                idx = i0 + j
                if idx >= N:
                    break
                save_path = out_root / cls_name / f"{idx:05d}.png"
                cv2.imwrite(str(save_path), im[:, :, ::-1])

        torch.distributed.barrier()
        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()
        total_time = (time.time() - time.time()) / 60
        print(f"Generated {args.num_images} images for each of {len(class_list)} classes in {total_time:.1f} min.")


if __name__ == '__main__':
    main()
