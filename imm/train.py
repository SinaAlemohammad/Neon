# train.py

import os
import re
import json
import hydra.types
import torch
import dnnlib
from training import training_loop
import warnings
from omegaconf import OmegaConf
import hydra

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 1.12.


@hydra.main(version_base=None, config_path="configs")
def main(cfg):

    config = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))
    
    # ---- Distributed training setup ----
    # Let torchrun handle spawning and environment variables.
    # The `main` function is now run once per process, so we initialize
    # the process group and device within this function.
    
    # Get rank and world_size from environment variables set by torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # We also need local_rank to correctly set the device
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Explicitly initialize with the NCCL backend for GPU training
        # We avoid the Gloo backend and the manual multiprocessing spawn
        # to prevent conflicts and ensure optimal performance.
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

        import torch.distributed as torch_dist
        if not torch_dist.is_initialized():
            torch_dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        from torch_utils import distributed as dist  # import after process group init
        
        # Set the device for this specific process
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
    else:
        # Fallback for non-distributed run
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        
        # For a single-process run, initialize dummy dist utility
        class DummyDist:
            def get_rank(self): return 0
            def get_world_size(self): return 1
            def print0(self, *args, **kwargs): print(*args, **kwargs)
        
        dist = DummyDist()

    # ---- The rest of your code remains largely the same ----

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**config.dataset)
        dataset_name = dataset_obj.name
        config.dataset.resolution = dataset_obj.resolution
        config.dataset.max_size = len(dataset_obj)
        del dataset_obj
    except IOError as err:
        raise ValueError(f"data: {err}")

    if config.augment:
        config.network.augment_dim = 9

    # Random seed.
    if config.training.seed is None:
        seed = torch.randint(1 << 31, size=[], device=device)
        if dist.get_world_size() > 1:
            torch.distributed.broadcast(seed, src=0)
        config.training.seed = int(seed)

    resume_tick = 0
    resume_pkl = resume_state_dump = None
    # Transfer learning and resume.
    if config.training.transfer is not None:
        if config.training.resume is not None:
            raise ValueError("--transfer and --resume cannot be specified at the same time")
        resume_pkl = config.training.transfer
        config.training.ema_rampup_ratio = None
    elif config.training.resume is not None:
        match = re.fullmatch(
            r"training-state-(\d+|latest).pt", os.path.basename(config.training.resume)
        )
        if not match or not os.path.isfile(config.training.resume):
            raise ValueError("--resume must point to training-state-*.pt from a previous training run")
        resume_pkl = os.path.join(
            os.path.dirname(config.training.resume),
            f"network-snapshot-{match.group(1)}.pkl",
        )
        resume_tick = (
            int(match.group(1))
            if config.training.resume_tick is None
            else config.training.resume_tick
        )
        resume_state_dump = config.training.resume

    # Description string.
    cond_str = "cond" if config.dataset.use_labels else "uncond"
    desc = f"{dataset_name:s}-{cond_str:s}-gpus{dist.get_world_size():d}"
    if config.name is not None:
        desc += f"-{config.name}"

    outdir = os.path.join(config.get('outputdir', 'outputs'), config.logger.project)

    # Pick output directory.
    if dist.get_rank() != 0:
        run_dir = None
    else:
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [
                x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))
            ]
        prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        run_dir = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")
        assert not os.path.exists(run_dir)

    # Print options.
    dist.print0()
    dist.print0("Training options:")
    dist.print0(json.dumps(OmegaConf.to_container(config), indent=2))
    dist.print0()
    dist.print0(f"Output directory:         {run_dir}")
    dist.print0(f"Dataset path:             {config.dataset.path}")
    dist.print0(f"Class-conditional:        {config.dataset.use_labels}")
    dist.print0(f"Number of GPUs:           {dist.get_world_size()}")
    dist.print0(f"Batch size:               {config.training.batch_size}")
    dist.print0(f"Mixed-precision:          {config.network.get('mixed_precision', 'fp32')}")
    dist.print0()

    # Create output directory.
    dist.print0("Creating output directory...")
    if dist.get_rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "training_options.json"), "wt") as f:
            json.dump(OmegaConf.to_container(config), f, indent=2)
        dnnlib.util.Logger(
            file_name=os.path.join(run_dir, "log.txt"),
            file_mode="a",
            should_flush=True,
        )

    # Train.
    training_loop.training_loop(
        config=config,
        resume_pkl=resume_pkl,
        resume_tick=resume_tick,
        resume_state_dump=resume_state_dump,
        run_dir=run_dir,
    )

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------