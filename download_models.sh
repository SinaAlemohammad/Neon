#!/usr/bin/env bash
set -euo pipefail

# Use faster transfer if available
export HF_HUB_ENABLE_HF_TRANSFER=1

# Ensure 'hf' CLI is available
if ! command -v hf >/dev/null 2>&1; then
  echo "Installing huggingface_hub (provides 'hf' CLI)..."
  pip install -U huggingface_hub >/dev/null
fi

mkdir -p checkpoints fid_stats

download() {
  local filename="$1"
  local outdir="$2"
  if [ -f "${outdir}/${filename}" ]; then
    echo "✓ ${outdir}/${filename} exists — skipping"
  else
    echo "Downloading ${filename} -> ${outdir}"
    hf download sinaalemohammad/Neon --include "${filename}" --local-dir "${outdir}"
  fi
}

# Models
download Neon_xARL_imagenet256.pth checkpoints
download Neon_xARB_imagenet256.pth checkpoints
download Neon_VARd16_imagenet256.pth checkpoints
download Neon_VARd36_imagenet512.pth checkpoints
download Neon_imm_imagenet256.pkl checkpoints
download Neon_EDM_conditional_CIFAR10.pkl checkpoints
download Neon_EDM_unconditional_CIFAR10.pkl checkpoints
download Neon_EDM_FFHQ.pkl checkpoints

# FID reference stats
download adm_in256_stats.npz fid_stats
download adm_in512_stats.npz fid_stats

echo "All downloads complete."
