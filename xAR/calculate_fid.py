#!/usr/bin/env python
# xAR/calculate_fid.py

import os
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")  
os.environ.setdefault("NCCL_DEBUG", "ERROR")           
os.environ.setdefault("PYTHONWARNINGS", "ignore")      
os.environ.setdefault("OMP_NUM_THREADS", "1")
# --------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")  # belt-and-suspenders


import argparse,shutil, random, contextlib
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch_fidelity
from torch.amp import autocast
from tqdm import tqdm

from util import misc
from models.vae import AutoencoderKL
from models import xar


def parse_args():
    p = argparse.ArgumentParser('Generate xAR samples + compute FID/IS (single model, quiet)')
    p.add_argument('--model', type=str, required=True, help='Which xar.* constructor to use')
    p.add_argument('--model_ckpt', type=str, required=True, help='Path to .pth weights (single model)')
    p.add_argument('--cfg', type=float, required=True, help='Classifier-free guidance scale')
    p.add_argument('--vae_path', type=str, required=True, help='KL-VAE checkpoint path')
    p.add_argument('--num_images', type=int, default=50000, help='Total images to generate')
    p.add_argument('--batch_size', type=int, default=64, help='Images per device per forward pass')
    p.add_argument('--flow_steps', type=int, default=64, help='Diffusion/flow steps')
    p.add_argument('--img_size', type=int, default=256, help='Image resolution for stats')
    p.add_argument('--class_num', type=int, default=1000, help='Number of classes')
    p.add_argument('--device', type=str, default='cuda', help='Torch device ("cuda" or "cpu")')
    p.add_argument('--fid_stats', type=str, default=None, help='Precomputed FID stats file (npz)')
    p.add_argument('--no_isc', action='store_true', help='Skip Inception Score')
    p.add_argument('--cuda', action='store_true', help='(ignored; metrics always on CUDA)')

    # DDP knobs
    p.add_argument('--dist_url', default='env://')
    p.add_argument('--world_size', type=int, default=1)
    p.add_argument('--local_rank', type=int, default=-1)
    p.add_argument('--dist_on_itp', action='store_true', default=False)

    p.add_argument('--out_dir', type=str, default='temp_images', help='Temporary folder for generated images (deleted)')
    p.add_argument('--seed', type=int, default=1, help='Random seed')
    return p.parse_args()


def load_checkpoint(path):
    # Prefer safe weight-only load if available
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def prepare_labels(num_images, class_num, batch_size, world):
    per = num_images // class_num
    if per == 0:
        raise ValueError(f"num_images ({num_images}) must be >= class_num ({class_num}).")
    labels = torch.arange(class_num, dtype=torch.long).repeat_interleave(per)
    bs_world = batch_size * world
    rem = labels.numel() % bs_world
    if rem:
        pad = bs_world - rem
        labels = torch.cat([labels, torch.zeros(pad, dtype=torch.long)], 0)
    return labels, labels.numel() // bs_world, bs_world


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def main():
    args = parse_args()

    # Enforce CUDA metrics for torch_fidelity (as requested)
    args.cuda = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for torch_fidelity, but no CUDA device is available.")

    # Establish local rank & select CUDA device BEFORE DDP init (prevents NCCL warnings)
    if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available() and args.local_rank is not None and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    # Quiet distributed init
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        misc.init_distributed_mode(args)

    rank, world = misc.get_rank(), misc.get_world_size()

    # Device-aware barrier helper (no NCCL "devices unknown" spam)
    def ddp_barrier():
        if dist.is_available() and dist.is_initialized():
            if dist.get_backend() == "nccl":
                dev_id = args.local_rank if (args.local_rank is not None and args.local_rank >= 0) else 0
                dist.barrier(device_ids=[dev_id])
            else:
                dist.barrier()

    # Deterministic but distinct across ranks
    set_seed(args.seed + max(rank, 0))

    out_root = Path(args.out_dir)
    model = net = vae = None  # for safe teardown

    try:
        if rank == 0:
            shutil.rmtree(out_root, ignore_errors=True)
            out_root.mkdir(parents=True, exist_ok=True)

        # ---------- VAE ----------
        vae = AutoencoderKL(embed_dim=16, ch_mult=(1,1,2,2,4), ckpt_path=args.vae_path)
        vae.eval().to(args.device)
        for p in vae.parameters():
            p.requires_grad = False

        # ---------- xAR ----------
        assert args.model in xar.__dict__, f"Model '{args.model}' not found in models.xar"
        net = xar.__dict__[args.model](
            img_size=args.img_size, vae_stride=16, patch_size=1,
            vae_embed_dim=16, class_num=args.class_num,
            attn_dropout=0.1, proj_dropout=0.1
        ).to(args.device)

        ckpt = load_checkpoint(args.model_ckpt)
        state = ckpt.get('model_ema', ckpt.get('state_dict', ckpt))
        net.load_state_dict(state, strict=True)
        net.eval()

        model = (torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank], output_device=args.local_rank
        ) if world > 1 else net)

        # ---------- labels ----------
        cfg = args.cfg
        labels, total_batches, bs_world = prepare_labels(args.num_images, args.class_num, args.batch_size, world)

        tag = f"eval_{Path(args.model_ckpt).stem}_cfg{cfg:.3f}"
        img_dir = out_root / tag
        if rank == 0:
            img_dir.mkdir(parents=True, exist_ok=True)
        ddp_barrier()

        # ---------- sampling ----------
        is_cuda_dev = (torch.cuda.is_available() and 'cuda' in str(args.device))
        autocast_ctx = (lambda: autocast(device_type='cuda')) if is_cuda_dev else contextlib.nullcontext

        it = tqdm(range(total_batches), desc=f"Gen-{tag}", dynamic_ncols=True, leave=False) if rank == 0 \
             else range(total_batches)

        for b in it:
            idx0 = b * bs_world + rank * args.batch_size
            if idx0 >= args.num_images:
                break
            idx1 = idx0 + args.batch_size
            lbl = labels[idx0:idx1].to(args.device)
            with autocast_ctx():
                sampler = model.module.sample_tokens if world > 1 else model.sample_tokens
                z = sampler(num_steps=args.flow_steps, cfg=cfg, label=lbl)
                imgs = vae.decode(z / 0.2325)
                imgs = (imgs + 1) / 2
            arr = imgs.mul(255).clamp_(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
            for j, im in enumerate(arr):
                gid = idx0 + j
                if gid >= args.num_images:
                    break
                cv2.imwrite(str(img_dir / f"{gid:05d}.png"), im[:, :, ::-1])

        ddp_barrier()

        # ---------- metrics (rank 0) ----------
        if rank == 0:
            stats = args.fid_stats or ('fid_stats/adm_in256_stats.npz' if args.img_size == 256 else None)
            if stats is None:
                raise NotImplementedError(f"No default FID stats for img_size={args.img_size}. Pass --fid_stats.")
            m = torch_fidelity.calculate_metrics(
                input1=str(img_dir), input2=None,
                fid_statistics_file=stats, cuda=True,  # enforce CUDA
                isc=not args.no_isc, fid=True, kid=False, prc=False, verbose=False
            )
            fid_val = float(m['frechet_inception_distance'])
            isc_mean = m.get('inception_score_mean')
            isc_std  = m.get('inception_score_std')

            parts = [f"FID={fid_val:.4f}"]
            if isc_mean is not None and isc_std is not None:
                parts.append(f"IS={isc_mean:.4f}±{isc_std:.4f}")
            parts += [
                f"cfg={cfg:g}",
                f"steps={args.flow_steps}",
                f"n={args.num_images}",
                f"model={args.model}",
                f"ckpt={Path(args.model_ckpt).stem}"
            ]
            print(" | ".join(parts))

            # cleanup images + temp root
            shutil.rmtree(img_dir, ignore_errors=True)
            try:
                out_root.rmdir()
            except OSError:
                pass

    finally:
        # --- graceful teardown ---
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        with contextlib.suppress(Exception):
            del model, net, vae  # free DDP refs first

        if dist.is_available() and dist.is_initialized():
            try:
                if dist.get_backend() == "nccl":
                    dev_id = args.local_rank if (args.local_rank is not None and args.local_rank >= 0) else 0
                    dist.barrier(device_ids=[dev_id])
                else:
                    dist.barrier()
            except Exception:
                pass
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
        # --------------------------------------------------------------------


if __name__ == '__main__':
    main()
