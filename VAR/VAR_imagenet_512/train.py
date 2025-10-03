#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import math
from functools import partial
from itertools import islice
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler
from utils.misc import auto_resume

# This script trains until a specified number of images seen, with linear warmup to a target LR.

def build_everything(args: arg_util.Args):
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-crypt*.pth')

    if dist.is_master():
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(
                log_dir=args.tb_log_dir_path,
                filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'
            ), verbose=True
        )
        tb_lg.flush()
    else:
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()

    if not args.local_debug:
        num_classes, dataset_train, _ = build_dataset(
            args.data_path, final_reso=args.data_load_reso,
            hflip=args.hflip, mid_reso=args.mid_reso,
        )
        dl = DataLoader(
            dataset=dataset_train,
            num_workers=args.workers,
            pin_memory=True,
            generator=args.get_different_generator_for_each_rank(),
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train),
                glb_batch_size=args.glb_batch_size,
                same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True,
                fill_last=True,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                start_ep=start_ep,
                start_it=start_it,
            ),
        )
        del dataset_train
        iters_train = len(dl)
        dl = iter(dl)
    else:
        num_classes, iters_train, dl = 1000, 10, None

    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import VAR, VQVAE, build_vae_var
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    print(args.depth)
    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=dist.get_device(), patch_nums=(1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
        num_classes=num_classes, depth=args.depth,
        shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng,
        init_head=args.hd, init_std=args.ini,
    )
    vae_ckpt = 'vae_ch160v4096z32.pth'
    if dist.is_local_master() and not os.path.exists(vae_ckpt):
        os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

    vae_local = args.compile_model(vae_local, args.vfast)
    var_wo_ddp.load_state_dict(torch.load('var_d36.pth', map_location='cpu'))
    var_wo_ddp = args.compile_model(var_wo_ddp, args.tfast)
    var = (
        DDP(var_wo_ddp, device_ids=[dist.get_local_rank()],
            find_unused_parameters=False, broadcast_buffers=False)
        if dist.initialized() else misc.NullDDP(var_wo_ddp)
    )

    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token','start_token','task_token','cfg_uncond',
        'pos_embed','pos_1LC','pos_start','start_pos','lvl_embed',
        'gamma','beta','ada_gss','moe_bias','scale_mul',
    })
    opt_clz = {
        'adam': partial(torch.optim.AdamW, betas=(0.9,0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9,0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=opt_clz(params=para_groups, lr=args.tlr, weight_decay=0),
        names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )

    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums,
        resos=args.resos, vae_local=vae_local,
        var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
    )
    if trainer_state:
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)
    dist.barrier()

    return tb_lg, trainer, start_ep, start_it, iters_train, dl

def train_one_ep(ep, is_first_ep, start_it, args, tb_lg, dl, iters_train, trainer):
    show_bar = dist.is_master()
    if show_bar:
        batch_bar = tqdm(total=iters_train,
                         desc=f"Batch ({ep+1}/{args.ep})",
                         unit="batch", position=1, leave=False)

    args.next_save_images = getattr(args, 'next_save_images', 0)
    for _ in range(start_it):
        next(dl)

    for it in range(start_it, iters_train):
        inp, label = next(dl)
        if show_bar:
            batch_bar.update(1)

        g_it = ep * iters_train + it
        images_seen = g_it * args.glb_batch_size

        # stop after total_images
        if args.total_images and images_seen >= args.total_images:
            if show_bar:
                batch_bar.close()
            return True

        # checkpoint by images_seen
        if images_seen >= args.next_save_images:
            if dist.is_local_master():
                os.makedirs(args.local_out_dir_path, exist_ok=True)
                path = os.path.join(args.local_out_dir_path,
                                    f'ar-ckpt-images_{images_seen}.pth')
                torch.save(trainer.var_wo_ddp.state_dict(), path)
            dist.barrier()
            args.next_save_images += args.save_interval

        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        args.cur_it = f"{it+1}/{iters_train}"

        # linear warmup to target_lr
        target = args.target_lr if args.target_lr is not None else args.tlr
        cur_lr = (target * min(images_seen, args.total_images) / args.total_images
                  ) if args.total_images > 0 else target
        for pg in trainer.var_opt.optimizer.param_groups:
            pg['lr'] = cur_lr
        args.cur_lr = cur_lr
        if show_bar:
            batch_bar.set_postfix({'lr': f"{cur_lr:.2e}"})

        # step
        stepping = (g_it + 1) % args.ac == 0
        trainer.train_step(
            it=it, g_it=g_it, stepping=stepping,
            metric_lg=misc.MetricLogger(delimiter='  '),
            tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label,
            prog_si=-1, prog_wp_it=args.pgwp * iters_train,
        )

    if show_bar:
        batch_bar.close()
    return False

def main_training():
    args = arg_util.init_dist_and_get_args()
    # sanity check prints
    print(f"[Sanity] total_images = {args.total_images}, save_interval = {args.save_interval}, target_lr = {args.target_lr}")
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)


    tb_lg, trainer, start_ep, start_it, iters_train, dl = build_everything(args)

    # initial checkpoint
    if dist.is_local_master():
        os.makedirs(args.local_out_dir_path, exist_ok=True)
        torch.save(trainer.var_wo_ddp.state_dict(),
                   os.path.join(args.local_out_dir_path, 'ar-ckpt-images_0.pth'))
    dist.barrier()

    # compute epochs needed
    print(f'global batch size is {args.glb_batch_size}')
    
    if args.total_images > 0:
        images_per_epoch = iters_train * args.glb_batch_size
        seen = start_ep * images_per_epoch + start_it * args.glb_batch_size
        rem = args.total_images - seen
        if rem <= 0:
            print(f"Already seen {seen} images ≥ target {args.total_images}. Exiting.")
            return
        args.ep = start_ep + math.ceil(rem / images_per_epoch)
        print(f"Training until {args.total_images} images seen: {args.ep - start_ep} more epochs (total {args.ep})")

    show_bar = dist.is_master()
    if show_bar:
        epoch_bar = tqdm(total=args.ep - start_ep, desc="Epoch", unit="ep", position=0)

    stop = False
    for ep in range(start_ep, args.ep):
        if hasattr(dl, 'sampler'):
            dl.sampler.set_epoch(ep)
        tb_lg.set_step(ep * iters_train)

        if train_one_ep(ep, ep == start_ep,
                        start_it if ep == start_ep else 0,
                        args, tb_lg, dl, iters_train, trainer):
            stop = True

        if show_bar:
            epoch_bar.update(1)
        tb_lg.flush()
        if stop:
            break

    if show_bar:
        epoch_bar.close()

if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close()
            sys.stderr.close()
