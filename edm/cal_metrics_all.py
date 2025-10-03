import os
import re
import glob
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

#----------------------------------------------------------------------------
# Sampler Function (EDM Sampler Only)
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

#----------------------------------------------------------------------------
# Helper: Stacked Random Generator
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32))
                           for seed in seeds]
    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs)
                            for gen in self.generators])
    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)
    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                            for gen in self.generators])

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Custom Dataset for In-Memory Generated Images
class GeneratedImagesDataset(Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 3 and img.shape[0] != 3:
            img = np.transpose(img, (2, 0, 1))
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        return torch.from_numpy(img).to(torch.uint8), 0

#----------------------------------------------------------------------------
# Distributed In-Memory FID Calculation
def calculate_inception_stats_from_dataset(dataset, detector_net, batch_size=64, num_workers=0, device=torch.device('cuda')):
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                             sampler=sampler, num_workers=num_workers)
    partial_mu    = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    partial_sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    partial_count = 0
    for images, _ in tqdm.tqdm(data_loader, unit='batch', disable=True):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = images.to(device)
        features = detector_net(images, **detector_kwargs).to(torch.float64)
        partial_mu    += features.sum(0)
        partial_sigma += features.T @ features
        partial_count += features.shape[0]
    if dist.is_initialized():
        count_tensor = torch.tensor(partial_count, dtype=torch.float64, device=device)
        dist.all_reduce(count_tensor)
        total_count   = count_tensor.item()
        dist.all_reduce(partial_mu)
        dist.all_reduce(partial_sigma)
    else:
        total_count = partial_count
    global_mu    = partial_mu / total_count
    global_sigma = partial_sigma / (total_count - 1) - torch.ger(global_mu, global_mu)
    return global_mu.cpu().numpy(), global_sigma.cpu().numpy()

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - 2 * s)
    return float(np.real(fid))

#----------------------------------------------------------------------------
# Function to Generate Images In-Memory (Distributed Version)
def generate_images(net, seeds, max_batch_size, class_idx, sampler_kwargs, device=torch.device('cuda')):
    rank       = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    seeds_list = parse_int_list(seeds) if isinstance(seeds, str) else seeds
    local_seeds = seeds_list[rank::world_size]
    images_list = []
    for i in tqdm.tqdm(range(0, len(local_seeds), max_batch_size),
                       desc='Generating batches', disable=(rank != 0)):
        batch_seeds = local_seeds[i:i+max_batch_size]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(batch_seeds)], device=device)]
        if class_idx is not None:
            class_labels = torch.zeros([len(batch_seeds), net.label_dim], device=device)
            class_labels[:, class_idx] = 1
        with torch.no_grad():
            images = edm_sampler(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        images_np = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        images_list.extend(images_np)
    return images_list

#----------------------------------------------------------------------------
# Unified Distributed Click Command
@click.command()
@click.option('--network_pkl_base',    help='Base network pickle filename (PATH or URL)', required=True)
@click.option('--network_pkl_aux_dir', help='Directory containing auxiliary network pickles', required=True)
@click.option('--differential_weights',help='Comma-separated list of differential weights', required=True)
@click.option('--max_training_step',   help='Max training step to process', type=int, default=10000)
@click.option('--seeds',               help='Random seeds for final eval (e.g. "1,2,5-10")', default='0-49999')
@click.option('--seeds_test',          help='Random seeds for test eval (e.g. "1,2,5-10")', default='0-9999')
@click.option('--max_batch_size',      help='Maximum batch size', type=int, default=64)
@click.option('--class', 'class_idx',  help='Class label (default: random)', type=int, default=None)
@click.option('--num_steps',           help='Number of sampling steps', type=int, default=18)
@click.option('--sigma_min',           help='Lowest noise level', type=float, default=0.002)
@click.option('--sigma_max',           help='Highest noise level', type=float, default=80)
@click.option('--rho',                 help='Time step exponent', type=float, default=7)
@click.option('--s_churn',             help='Stochasticity strength', type=float, default=0)
@click.option('--s_min',               help='Stoch. min noise level', type=float, default=0)
@click.option('--s_max',               help='Stoch. max noise level', type=float, default=float('inf'))
@click.option('--s_noise',             help='Stoch. noise inflation', type=float, default=1)
@click.option('--ref',                 help='Reference NPZ for FID stats', required=True)
@click.option('--out_dir',             help='Output directory for saving FID results', required=True)
def main(network_pkl_base, network_pkl_aux_dir, differential_weights,
         max_training_step, seeds, seeds_test, max_batch_size, class_idx,
         num_steps, sigma_min, sigma_max, rho,
         s_churn, s_min, s_max, s_noise, ref, out_dir):

    # Initialize distributed
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    # Load base network
    with dnnlib.util.open_url(network_pkl_base, verbose=False) as f:
        a_base     = pickle.load(f)
    model_base = a_base['ema'].to(device)
    base_state = model_base.state_dict()

    # Load Inception detector
    detector_url = (
        'https://api.ngc.nvidia.com/v2/models/nvidia/research/'
        'stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    )
    with dnnlib.util.open_url(detector_url, verbose=False) as f:
        detector_net = pickle.load(f).to(device)

    # Gather eligible aux files
    aux_pattern = os.path.join(network_pkl_aux_dir, "network-snapshot-*.pkl*")
    aux_files   = sorted(
        glob.glob(aux_pattern),
        key=lambda x: int(re.search(r'network-snapshot-(\d+)', os.path.basename(x)).group(1))
    )
    eligible_aux_files = [
        f for f in aux_files
        if int(re.search(r'network-snapshot-(\d+)', os.path.basename(f)).group(1)) <= max_training_step
    ]

    if rank == 0:
        click.echo("Eligible auxiliary snapshots:")
        for f in eligible_aux_files:
            click.echo("  " + os.path.basename(f))
    dist.barrier()

    # Parse differential weights
    differential_weights_list = [float(w.strip()) for w in differential_weights.split(',')]

    # Loop over aux snapshots
    for aux_file in eligible_aux_files:
        snapshot_num = re.search(r'network-snapshot-(\d+)', os.path.basename(aux_file)).group(1)

        # Skip snapshot 000050 entirely
        if snapshot_num == "000050":
            if rank == 0:
                click.echo(f"Skipping snapshot {snapshot_num}")
            continue

        if rank == 0:
            os.makedirs(os.path.join(out_dir, snapshot_num), exist_ok=True)
        dist.barrier()

        prev_fid = None
        threshold = 0.03
        best_fid = float('inf')
        best_w   = None

        # Loop over differential weights
        for diff_weight in differential_weights_list:
            # skip if output exists
            output_filename = os.path.join(out_dir, snapshot_num, f"w={diff_weight}.txt")
            skip = rank == 0 and os.path.exists(output_filename)
            skip_list = [skip]
            dist.broadcast_object_list(skip_list, src=0)
            if skip_list[0]:
                if rank == 0:
                    click.echo(f"Skipping {snapshot_num} @ w={diff_weight}: already exists")
                continue

            # Load and merge networks
            with dnnlib.util.open_url(aux_file, verbose=False) as f:
                a_aux = pickle.load(f)
            model_aux = a_aux['ema'].to(device)
            aux_state = model_aux.state_dict()
            with dnnlib.util.open_url(network_pkl_base, verbose=False) as f:
                temp = pickle.load(f)
            new_net = temp['ema'].to(device)
            new_state = new_net.state_dict()
            for key in new_state:
                if key in base_state and key in aux_state:
                    new_state[key] = (1 + diff_weight) * base_state[key] - diff_weight * aux_state[key]
                elif key in base_state:
                    new_state[key] = base_state[key]
                else:
                    new_state[key] = -diff_weight * aux_state.get(key, torch.zeros_like(new_state[key]))
            new_net.load_state_dict(new_state)
            net = new_net.eval()

            sampler_kwargs = {
                'num_steps': num_steps,
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'rho': rho,
                'S_churn': s_churn,
                'S_min': s_min,
                'S_max': s_max,
                'S_noise': s_noise,
            }

            # Generate & gather images (test seeds)
            images_local = generate_images(net, seeds_test, max_batch_size, class_idx, sampler_kwargs, device=device)
            all_images = [None] * world_size
            dist.all_gather_object(all_images, images_local)
            if rank == 0:
                merged_images = [img for sub in all_images for img in sub]
            else:
                merged_images = None
            merged_list = [merged_images]
            dist.broadcast_object_list(merged_list, src=0)
            merged_images = merged_list[0]

            # Compute stats
            dataset = GeneratedImagesDataset(merged_images)
            mu, sigma_val = calculate_inception_stats_from_dataset(
                dataset, detector_net, batch_size=max_batch_size, num_workers=0, device=device
            )

            # Rank 0: compute FID & threshold check
            stop_flag = False
            if rank == 0:
                with dnnlib.util.open_url(ref, verbose=False) as f:
                    ref_stats = dict(np.load(f))
                fid_value = calculate_fid_from_inception_stats(
                    mu, sigma_val, ref_stats['mu'], ref_stats['sigma']
                )
                click.echo(f"Aux {snapshot_num} @ w={diff_weight}: FID {fid_value:g}")

                if prev_fid is not None and fid_value > prev_fid + threshold:
                    click.echo(f"FID jumped by >{threshold:g} (from {prev_fid:g} to {fid_value:g}); stopping.")
                    stop_flag = True
                prev_fid = fid_value

                # track best
                if fid_value < best_fid:
                    best_fid = fid_value
                    best_w   = diff_weight

            # broadcast results
            fid_list = [None]
            stop_list = [None]
            if rank == 0:
                fid_list[0] = fid_value
                stop_list[0] = stop_flag
            dist.broadcast_object_list(fid_list, src=0)
            dist.broadcast_object_list(stop_list, src=0)
            fid_value = fid_list[0]
            stop_flag = stop_list[0]

            # save intermediate FID
            if rank == 0:
                with open(output_filename, 'w') as fp:
                    fp.write(str(fid_value))
            dist.barrier()

            if stop_flag:
                break

        # --- After differential_weights loop: recompute on training seeds at best_w ---
        best_w_list = [best_w]
        dist.broadcast_object_list(best_w_list, src=0)
        best_w = best_w_list[0]

        # reload & merge with best_w
        with dnnlib.util.open_url(aux_file, verbose=False) as f:
            a_aux = pickle.load(f)
        model_aux = a_aux['ema'].to(device)
        aux_state = model_aux.state_dict()
        with dnnlib.util.open_url(network_pkl_base, verbose=False) as f:
            temp = pickle.load(f)
        new_net = temp['ema'].to(device)
        new_state = new_net.state_dict()
        for key in new_state:
            if key in base_state and key in aux_state:
                new_state[key] = (1 + best_w) * base_state[key] - best_w * aux_state[key]
            elif key in base_state:
                new_state[key] = base_state[key]
            else:
                new_state[key] = -best_w * aux_state.get(key, torch.zeros_like(new_state[key]))
        new_net.load_state_dict(new_state)
        net = new_net.eval()

        # generate & gather images (training seeds)
        images_local = generate_images(net, seeds, max_batch_size, class_idx, sampler_kwargs, device=device)
        all_images = [None] * world_size
        dist.all_gather_object(all_images, images_local)
        if rank == 0:
            merged_images = [img for sub in all_images for img in sub]
        else:
            merged_images = None
        merged_list = [merged_images]
        dist.broadcast_object_list(merged_list, src=0)
        merged_images = merged_list[0]

        # compute stats & final FID
        dataset = GeneratedImagesDataset(merged_images)
        mu, sigma_val = calculate_inception_stats_from_dataset(
            dataset, detector_net, batch_size=max_batch_size, num_workers=0, device=device
        )
        if rank == 0:
            with dnnlib.util.open_url(ref, verbose=False) as f:
                ref_stats = dict(np.load(f))
            final_fid = calculate_fid_from_inception_stats(
                mu, sigma_val, ref_stats['mu'], ref_stats['sigma']
            )
            click.echo(f"Final FID on seeds @ w={best_w}: {final_fid:g}")
            best_outfile = os.path.join(out_dir, snapshot_num, f"best_w_{best_w}_fid.txt")
            with open(best_outfile, 'w') as fp:
                fp.write(str(final_fid))
        dist.barrier()

    # end aux_file loop

if __name__ == '__main__':
    main()