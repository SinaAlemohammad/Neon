import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder

from models.vae import AutoencoderKL
from models import xar

import copy

"""
Modified training script for xAR that saves checkpoints after a fixed number of
**images seen** instead of at fixed epoch intervals.

Key additions / changes
----------------------
1. **--save_img_interval** CLI argument (default **250_000** images).
2. `images_seen` counter is tracked globally across epochs and updated each
   iteration by `batch_size * world_size`.
3. When `images_seen` reaches or exceeds `next_chkpt`, rank‑0 process saves a
   checkpoint named `checkpoint-<images_seen>.pth` and schedules the next
   threshold.
4. `train_one_epoch` merged into this file (no separate engine_xar import).

All original functionality is preserved; epoch checkpoints (`--save_last_freq`,
`--epoch_save_freq`) are disabled because the new image‑based schedule
supersedes them.
"""

def get_args_parser():
    parser = argparse.ArgumentParser('xAR training with image‑based checkpointing', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # gpus)')
    parser.add_argument('--epochs', default=400, type=int)

    # Checkpointing by images
    parser.add_argument('--save_img_interval', default=250_000, type=int,
                        help='Save checkpoint every N images seen (global count)')

    # Model parameters
    parser.add_argument('--model', default='xar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='VAE checkpoint path')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier‑free guidance")
    parser.add_argument('--cfg_schedule', default="cosine", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=0, help='disabled (legacy)')
    parser.add_argument('--epoch_save_freq', type=int, default=0, help='disabled (legacy)')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser


# ------------------------------------------------
# Training utilities (merged from engine_xar.py)
# ------------------------------------------------
import math
import sys
from typing import Iterable
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    images_seen: int = 0,
                    next_chkpt: int = 0):
    """Single‑epoch training loop.

    Returns
    -------
    images_seen : int
        Updated global image count *after* this epoch.
    next_chkpt : int
        Next image‑count threshold at which to save.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 100

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir:', log_writer.log_dir)

    world_size = misc.get_world_size()
    rank0 = misc.is_main_process()

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # learning‑rate schedule (per‑iteration)
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples)
            x = posterior.sample().mul_(0.2325)  # normalise latent std to 1

        # forward / backward
        with torch.cuda.amp.autocast():
            loss = model(x, labels)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # ----------------------------------------------------------
        # GLOBAL IMAGE COUNTER & ON‑THE‑FLY CHECKPOINTING
        # ----------------------------------------------------------
        images_seen += args.batch_size * world_size
        if rank0 and images_seen >= next_chkpt:
            chkpt_name = f"checkpoint-{images_seen}"
            misc.save_model(args=args,
                            model=model,
                            model_without_ddp=(model.module if hasattr(model, 'module') else model),
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch,
                            ema_params=ema_params,
                            epoch_name=chkpt_name)
            print(f"[Checkpoint] Saved after {images_seen} images → {chkpt_name}")
            next_chkpt += args.save_img_interval
        # ----------------------------------------------------------

        # TensorBoard
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # end of epoch
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print(f"images seen = {images_seen/1000000}")
    return images_seen, next_chkpt

# ------------------------------------------------
# Main
# ------------------------------------------------

def main(args):
    misc.init_distributed_mode(args)

    print('job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(str(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # TensorBoard
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Augmentations (DiT / ADM)
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)

    # VAE + xAR model
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    for p in vae.parameters():
        p.requires_grad = False

    model = xar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
    )
    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params/1e6:.2f}M")

    model.to(device)

    eff_batch = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size:", eff_batch)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    model_without_ddp = model.module if hasattr(model, 'module') else model

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.96))
    loss_scaler = NativeScaler()

    # Resume
    images_seen = 0
    next_chkpt = args.save_img_interval  # first checkpoint → N images

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt)
        images_seen = ckpt.get('images_seen', 0)  # optional backward‑compat
        next_chkpt = ((images_seen // args.save_img_interval) + 1) * args.save_img_interval
        if 'optimizer' in ckpt and 'epoch' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = ckpt['epoch'] + 1
            if 'scaler' in ckpt:
                loss_scaler.load_state_dict(ckpt['scaler'])
        print("Resumed from", args.resume)
        del ckpt
    else:
        print("Training from scratch")

    model_params = [p for p in model_without_ddp.parameters()]
    ema_params = copy.deepcopy(model_params)

    # Training loop --------------------------------------------------
    print(f"Start training for {args.epochs} epochs (checkpoint every {args.save_img_interval} images)")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        images_seen, next_chkpt = train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            images_seen=images_seen,
            next_chkpt=next_chkpt)

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    print('Training time', str(datetime.timedelta(seconds=int(total_time))), f'images_seen={images_seen}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('xAR training wrapper', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
