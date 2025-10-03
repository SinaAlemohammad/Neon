#!/usr/bin/env python3
"""
Unified Distributed Script for Generating Images and Evaluating PRDC
(Precision, Recall, Density, Coverage) using:
  - Feature extraction from clean-fid (InceptionV3 pool3, 2048-D)
  - PRDC metric from the prdc library

No torch-fidelity is used.

Install deps:
  pip install torch torchvision clean-fid prdc pillow click tqdm scikit-learn
"""

import os
import re
import sys
import json
import math
import time
import click
import pickle
import datetime
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import dnnlib
from torch.utils.data import Dataset, DataLoader

# Feature extraction (InceptionV3 activations) from clean-fid
from cleanfid import fid
# PRDC metrics
from prdc import compute_prdc

# ──────────────────────────────────────────────────────────────────────────────
# EDM sampler
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=40, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1
):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if (S_min <= t_cur <= S_max) else 0
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

# ──────────────────────────────────────────────────────────────────────────────
# RNG + seed parsing
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(s) % (1 << 32)) for s in seeds]
    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=g, **kwargs) for g in self.generators])
    def randn_like(self, x):
        return self.randn(x.shape, dtype=x.dtype, layout=x.layout, device=x.device)
    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=g, **kwargs) for g in self.generators])

def parse_int_list(s):
    if isinstance(s, list):
        return s
    out = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in str(s).split(','):
        m = range_re.match(p.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Simple image writer for generated batches (distributed-safe)
def save_images_for_seeds(
    net, seeds, out_dir, max_batch_size, class_idx,
    sampler_kwargs, device
):
    """
    Generates images for the given seeds and saves them as PNG files into out_dir.
    Distributed-aware: each rank handles a strided subset of seeds and writes unique filenames.
    Shows a tqdm progress bar on rank 0; bar auto-clears when done.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    os.makedirs(out_dir, exist_ok=True)

    seeds_list = parse_int_list(seeds) if isinstance(seeds, str) else seeds
    local_seeds = seeds_list[rank::world]

    total_batches = (len(local_seeds) + max_batch_size - 1) // max_batch_size
    pbar = tqdm(total=total_batches, desc=f"Generating → {os.path.basename(out_dir)}",
                dynamic_ncols=True, leave=False) if rank == 0 else None

    for i in range(0, len(local_seeds), max_batch_size):
        batch_seeds = local_seeds[i:i + max_batch_size]
        if not batch_seeds:
            if pbar is not None: pbar.update(1)
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution],
            device=device
        )

        class_labels = None
        if getattr(net, "label_dim", 0):
            if class_idx is None:
                class_labels = torch.eye(net.label_dim, device=device)[
                    rnd.randint(net.label_dim, size=[len(batch_seeds)], device=device)
                ]
            else:
                class_labels = torch.zeros([len(batch_seeds), net.label_dim], device=device)
                class_labels[:, class_idx] = 1

        with torch.no_grad():
            images = edm_sampler(
                net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs
            )

        images_uint8 = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()
        images_uint8 = images_uint8.permute(0, 2, 3, 1).numpy()  # [B,H,W,C]

        for seed_val, img_np in zip(batch_seeds, images_uint8):
            fname = f"seed{int(seed_val):06d}-r{rank}.png"
            fpath = os.path.join(out_dir, fname)
            Image.fromarray(img_np).save(fpath, format="PNG")

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # one quick barrier **during generation** is fine (everyone is doing the same work)
    if dist.is_initialized():
        dist.barrier()

# ──────────────────────────────────────────────────────────────────────────────
# clean-fid API compatibility shims (handles old/new versions)
def _build_cf_model(cf_mode, cf_device, cf_model):
    """
    Build a clean-fid feature extractor across multiple possible signatures.
    Returns a callable model.
    """
    # Newer API
    try:
        return fid.build_feature_extractor(model_name=cf_model, mode=cf_mode, device=cf_device)
    except TypeError:
        pass
    # Mid API
    try:
        return fid.build_feature_extractor(mode=cf_mode, device=cf_device)
    except TypeError:
        pass
    # Oldest API (no args)
    return fid.build_feature_extractor()

def get_cf_features(path, cf_mode, feat_num_workers, feat_batch_size, cf_model=None, cf_device=None):
    """
    Returns numpy array [N, 2048] of Inception features for images under `path`.
    Builds the model explicitly to avoid NoneType call errors in older clean-fid.
    Tries multiple get_folder_features signatures.
    """
    model = _build_cf_model(cf_mode, cf_device, cf_model)

    # Try newest signature first (explicit model + device + loaders)
    try:
        return fid.get_folder_features(
            path,
            model=model,
            mode=cf_mode,
            num_workers=feat_num_workers,
            batch_size=feat_batch_size,
            device=cf_device,
        )
    except TypeError:
        pass

    # Try without device
    try:
        return fid.get_folder_features(
            path,
            model=model,
            mode=cf_mode,
            num_workers=feat_num_workers,
            batch_size=feat_batch_size,
        )
    except TypeError:
        pass

    # Oldest fallback (path, model)
    return fid.get_folder_features(path, model)

# ──────────────────────────────────────────────────────────────────────────────
def _bcast_bool(val: bool) -> bool:
    """
    Broadcast a boolean from rank 0 to all ranks and return the received value.
    """
    if not dist.is_initialized():
        return val
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([1 if val else 0], device=device, dtype=torch.int32)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())

# ──────────────────────────────────────────────────────────────────────────────
# CLI
@click.command()
@click.option('--network_pkl_base',      required=True, help='Base network pickle (PATH or URL)')
@click.option('--network_pkl_aux_file',  required=True, help='Aux network pickle (PATH or URL), must contain "network-snapshot-<step>" in filename')
@click.option('--differential_weights',  required=True, help='Comma-separated differential weights, e.g. "0,0.5,1.0"')
@click.option('--max_training_step',     type=int, default=3100, help='Max training step; aux snapshot must be <= this')
@click.option('--seeds',                 default='0-9999', help='Single seed set used for ALL weights (e.g., "0-9999")')
@click.option('--max_batch_size',        type=int, default=64, help='Max batch size for generation')
@click.option('--class', 'class_idx',    type=int, default=None, help='Class index if conditional; default: random labels')
@click.option('--num_steps',             type=int, default=18,  help='EDM sampling steps')
@click.option('--sigma_min',             type=float, default=0.002)
@click.option('--sigma_max',             type=float, default=80.0)
@click.option('--rho',                   type=float, default=7.0)
@click.option('--s_churn',               type=float, default=0.0)
@click.option('--s_min',                 type=float, default=0.0)
@click.option('--s_max',                 type=float, default=float('inf'))
@click.option('--s_noise',               type=float, default=1.0)
@click.option('--real_data_dir',         required=True, help='Directory of REAL images for PRDC features')
@click.option('--out_dir',               required=True, help='Directory to save PRDC JSONs')
@click.option('--gen_dir',               default='temp_images', help='Directory to store generated PNGs (kept)')
# PRDC + clean-fid knobs
@click.option('--nearest_k',             type=int, default=5, help='k for PRDC (common: 3 or 5)')
@click.option('--feat_batch_size',       type=int, default=128, help='Batch size for clean-fid feature extraction')
@click.option('--feat_num_workers',      type=int, default=4, help='DataLoader workers for clean-fid feature extraction')
@click.option('--cf_mode',               default='clean', type=click.Choice(['clean','legacy'], case_sensitive=False), help='clean-fid mode')
@click.option('--cf_model',              type=str, default='inception_v3', help='clean-fid model name (default inception_v3)')
@click.option('--poll_timeout_sec',      type=int, default=36000, help='Max seconds nonzero ranks wait for rank-0 flag')
def main(
    network_pkl_base, network_pkl_aux_file, differential_weights, max_training_step,
    seeds, max_batch_size, class_idx, num_steps, sigma_min, sigma_max, rho, s_churn, s_min, s_max, s_noise,
    real_data_dir, out_dir, gen_dir,
    nearest_k, feat_batch_size, feat_num_workers, cf_mode, cf_model, poll_timeout_sec
):
    # Device first (reduces NCCL warnings)
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Distributed init with generous timeout
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=12))
    rank = dist.get_rank()

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    cf_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base net (for merging)
    with dnnlib.util.open_url(network_pkl_base, verbose=False) as f:
        a_base = pickle.load(f)
    model_base = a_base['ema'].to(device).eval()
    base_state = model_base.state_dict()

    # Validate aux file (and step)
    aux_basename = os.path.basename(network_pkl_aux_file)
    m = re.search(r'network-snapshot-(\d+)', aux_basename)
    if m is None:
        if rank == 0:
            raise click.UsageError(f'--network_pkl_aux_file must include "network-snapshot-<step>" in its name; got: {aux_basename}')
        return
    snapshot_int = int(m.group(1))
    if snapshot_int > max_training_step:
        if rank == 0:
            click.echo(f"Aux file {aux_basename} (snapshot {snapshot_int}) exceeds --max_training_step {max_training_step}; exiting.")
        return

    # Parse weights
    diff_ws = [float(w.strip()) for w in str(differential_weights).split(',') if w.strip()]

    # Sampler kwargs
    sampler_kwargs = dict(
        num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
        rho=rho, S_churn=s_churn, S_min=s_min, S_max=s_max, S_noise=s_noise
    )

    # Loop per differential weight
    for w in diff_ws:
        # Filenames for this weight
        w_dir_name = f"w_{w}".replace('.', 'p').replace('-', 'm')
        gen_subdir = os.path.join(gen_dir, w_dir_name)
        done_flag = os.path.join(out_dir, f".done_{w_dir_name}")
        out_json = os.path.join(out_dir, f"prdc_{w_dir_name}.json")

        # Rank 0 checks if PR/RC already exist; broadcast decision
        if rank == 0:
            skip = False
            if os.path.exists(out_json):
                try:
                    with open(out_json, "r") as f:
                        data = json.load(f)
                    # consider it valid if precision & recall keys exist
                    if "precision" in data and "recall" in data:
                        skip = True
                        print(f"[w={w}] Found existing PRDC at {out_json} (P={data.get('precision'):.6f}, R={data.get('recall'):.6f}); skipping generation & PRDC.", flush=True)
                except Exception:
                    skip = False
        else:
            skip = False
        skip = _bcast_bool(skip)

        if skip:
            # Ensure other ranks don't wait on a flag for this weight
            continue

        # Merge: (1+w)*base - w*aux
        with dnnlib.util.open_url(network_pkl_aux_file, verbose=False) as f:
            a_aux = pickle.load(f)
        model_aux = a_aux['ema'].to(device).eval()
        aux_state = model_aux.state_dict()

        with dnnlib.util.open_url(network_pkl_base, verbose=False) as f:
            tmp = pickle.load(f)
        new_net = tmp['ema'].to(device).eval()
        new_state = new_net.state_dict()

        for k in new_state:
            if k in base_state and k in aux_state:
                new_state[k] = (1.0 + w) * base_state[k] - w * aux_state[k]
            elif k in base_state:
                new_state[k] = base_state[k]
            else:
                new_state[k] = -w * aux_state.get(k, torch.zeros_like(new_state[k]))
        new_net.load_state_dict(new_state)
        net = new_net.eval()

        # 1) All ranks generate & save images
        save_images_for_seeds(
            net=net, seeds=seeds, out_dir=gen_subdir, max_batch_size=max_batch_size,
            class_idx=class_idx, sampler_kwargs=sampler_kwargs, device=device
        )

        # 2) Rank 0: extract features with clean-fid and compute PRDC; others wait
        if rank == 0:
            # clear any stale flag
            if os.path.exists(done_flag):
                try: os.remove(done_flag)
                except OSError: pass

            # Feature extraction with clean-fid (returns numpy array [N,2048])
            feats_fake = get_cf_features(
                gen_subdir, cf_mode, feat_num_workers, feat_batch_size, cf_model=cf_model, cf_device=cf_device
            )
            feats_real = get_cf_features(
                real_data_dir, cf_mode, feat_num_workers, feat_batch_size, cf_model=cf_model, cf_device=cf_device
            )

            # Compute PRDC
            prdc = compute_prdc(
                real_features=feats_real,
                fake_features=feats_fake,
                nearest_k=nearest_k
            )
            # Ensure floats for JSON
            prdc_json = {k: float(v) for k, v in prdc.items()}

            # Save metrics JSON
            with open(out_json, "w") as f:
                json.dump(prdc_json, f, indent=2)

            print(
                f"[w={w}] "
                f"P={prdc_json['precision']:.6f}  "
                f"R={prdc_json['recall']:.6f}  "
                f"D={prdc_json['density']:.6f}  "
                f"C={prdc_json['coverage']:.6f}  -> {out_json}",
                flush=True
            )

            # Signal completion to other ranks
            with open(done_flag, "w") as f:
                f.write("ok\n")
        else:
            # Non-zero ranks: wait for completion flag (no NCCL ops here)
            start = time.time()
            while not os.path.exists(done_flag):
                if time.time() - start > poll_timeout_sec:
                    raise RuntimeError(f"Rank-0 did not complete PRDC for {w_dir_name} in time.")
                time.sleep(2.0)

        # proceed to next weight (no barrier here)

    # Keep generated images for inspection/reuse

    # Safely tear down the process group for all ranks and exit
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    sys.exit(0)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
