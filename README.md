# Neon — Post‑hoc Self-Training via Reverse Weight Merging

Neon is a **one-line, post-hoc** trick to improve a trained generator **without new real data**. You briefly **self-train on your model’s own samples** (which nudges it in a *mode-seeking* direction) and then **merge weights in the *opposite* direction** of that update to correct the bias. In practice this **recovers recall and lowers FID** for AR, diffusion/flow, and few-step samplers with minimal extra compute.

## Method (concise)
- **Setup.** Let \( \theta_r \) be a trained base model. Sample a synthetic set \(S\) using your *usual inference* (e.g., CFG, finite-step ODE/flow, or low-temperature decoding).
- **Brief self-train.** Fine-tune on \(S\) → get \( \theta_s \). This update typically drifts toward over-confident, mode-heavy regions.
- **Reverse merge (Neon).** Form the final checkpoint by reversing that drift with a scalar \( w>0 \):  
  \[ \theta_{\text{neon}} = (1+w)\,\theta_r - w\,\theta_s. \]
  Tuning \(w\) on a small validation grid is usually enough (often unimodal behavior).
- **When it helps.** With **mode-seeking** inference (CFG, finite-step solvers, low-temp/top-k/p decoding), the synthetic gradient is *anti-aligned* with the population gradient; flipping it corrects bias → **better recall / FID**.
- **When to skip.** If you intentionally use **diversity-seeking** sampling (e.g., AR with very high temperature), *vanilla* self-training may be preferable.

---

## Quickstart

### 1) Environment
```bash
# from repo root
conda env create -f environment.yml
conda activate neon
```

### 2) Download checkpoints & FID stats
To keep things short here, just run:
```bash
bash download_models.sh
```
This will populate `checkpoints/` and `fid_stats/` with the required files.

### 3) Evaluate (FID/IS)
Below are quiet, print-only commands that generate in memory and report FID/IS. All assume **8 GPUs**; adjust `--nproc_per_node` as needed.

**xAR VAE dependency (credit: [MAR](https://github.com/LTH14/mar))**  
Before running xAR FID, download the KL‑16 VAE checkpoint to the existing folder:
```bash
hf download xwen99/mar-vae-kl16 --include kl16.ckpt --local-dir xAR/pretrained
```
Use it via:
```bash
--vae_path xAR/pretrained/kl16.ckpt
```

**xAR (ImageNet-256)**
```bash
# Large (xAR-L)
PYTHONPATH=xAR torchrun --standalone --nproc_per_node=8 xAR/calculate_fid.py   --model xar_large   --model_ckpt checkpoints/Neon_xARL_imagenet256.pth   --cfg 2.3   --vae_path xAR/pretrained/kl16.ckpt   --num_images 50000 --batch_size 64 --flow_steps 40 --img_size 256   --fid_stats fid_stats/adm_in256_stats.npz

# Base (xAR-B)
PYTHONPATH=xAR torchrun --standalone --nproc_per_node=8 xAR/calculate_fid.py   --model xar_base   --model_ckpt checkpoints/Neon_xARB_imagenet256.pth   --cfg 2.7   --vae_path xAR/pretrained/kl16.ckpt   --num_images 50000 --batch_size 32 --flow_steps 50 --img_size 256   --fid_stats fid_stats/adm_in256_stats.npz
```

**VAR (ImageNet-256 / 512)**
```bash
# d16 @ 256
PYTHONPATH=VAR/VAR_imagenet_256 torchrun --standalone --nproc_per_node=8 VAR/VAR_imagenet_256/calculate_fid.py   --var_ckpt checkpoints/Neon_VARd16_imagenet256.pth   --num_images 50000 --batch_size 64 --img_size 256   --fid_stats fid_stats/adm_in256_stats.npz

# d36 @ 512
PYTHONPATH=VAR/VAR_imagenet_512 torchrun --standalone --nproc_per_node=8 VAR/VAR_imagenet_512/calculate_fid.py   --var_ckpt checkpoints/Neon_VARd36_imagenet512.pth   --num_images 50000 --batch_size 32 --img_size 512   --fid_stats fid_stats/adm_in512_stats.npz
```

**EDM (CIFAR-10, FFHQ)**
```bash
# CIFAR-10 (conditional)
PYTHONPATH=edm torchrun --standalone --nproc_per_node=8 edm/calculate_fid.py   --network_pkl checkpoints/Neon_EDM_conditional_CIFAR10.pkl   --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz   --seeds 0-49999 --max_batch_size 256 --num_steps 18

# CIFAR-10 (unconditional)
PYTHONPATH=edm torchrun --standalone --nproc_per_node=8 edm/calculate_fid.py   --network_pkl checkpoints/Neon_EDM_unconditional_CIFAR10.pkl   --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz   --seeds 0-49999 --max_batch_size 256 --num_steps 18

# FFHQ-64×64 (unconditional)
PYTHONPATH=edm torchrun --standalone --nproc_per_node=8 edm/calculate_fid.py   --network_pkl checkpoints/Neon_EDM_FFHQ.pkl   --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz   --seeds 0-49999 --max_batch_size 256 --num_steps 40
```

**IMM (ImageNet-256)**
```bash
# Single-model FID/IS
PYTHONPATH=imm torchrun --standalone --nproc_per_node=8 imm/calculate_fid.py   --network_pkl checkpoints/Neon_imm_imagenet256.pkl   --fid_stats fid_stats/adm_in256_stats.npz   --num_steps 8 --cfg_scale 1.8
```

---

## Checkpoints (links & reported FID)

| Model type | Dataset | Link to download | FID (paper) |
|---|---|---|---|
| xAR-L | ImageNet-256 | [Neon_xARL_imagenet256.pth](https://huggingface.co/sinaalemohammad/Neon/resolve/main/Neon_xARL_imagenet256.pth) | **1.02** |
| xAR-B | ImageNet-256 | [Neon_xARB_imagenet256.pth](https://huggingface.co/sinaalemohammad/Neon/resolve/main/Neon_xARB_imagenet256.pth) | TBD |
| VAR d16 | ImageNet-256 | [Neon_VARd16_imagenet256.pth](https://huggingface.co/sinaalemohammad/Neon/resolve/main/Neon_VARd16_imagenet256.pth) | **2.01** |
| VAR d36 | ImageNet-512 | [Neon_VARd36_imagenet512.pth](https://huggingface.co/sinaalemohammad/Neon/resolve/main/Neon_VARd36_imagenet512.pth) | TBD |
| EDM (cond.) | CIFAR-10 (32×32) | [Neon_EDM_conditional_CIFAR10.pkl](https://huggingface.co/sinaalemohammad/Neon/resolve/main/Neon_EDM_conditional_CIFAR10.pkl) | **1.38** |
| EDM (uncond.) | CIFAR-10 (32×32) | [Neon_EDM_unconditional_CIFAR10.pkl](https://huggingface.co/sinaalemohammad/Neon/resolve/main/Neon_EDM_unconditional_CIFAR10.pkl) | TBD |
| EDM | FFHQ-64×64 | [Neon_EDM_FFHQ.pkl](https://huggingface.co/sinaalemohammad/Neon/resolve/main/Neon_EDM_FFHQ.pkl) | **1.12** |
| IMM | ImageNet-256 | [Neon_imm_imagenet256.pkl](https://huggingface.co/sinaalemohammad/Neon/resolve/main/Neon_imm_imagenet256.pkl) | TBD |

---

## License
MIT (see `LICENSE`).

## Contact
Questions or issues? Open a GitHub issue or reach out to **@sinaalemohammad**.
