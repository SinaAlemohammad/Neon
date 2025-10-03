#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export LOGLEVEL=ERROR
export PYTHONWARNINGS=ignore

############## |S| = 750k, Var on imagenet 256 #############################

#### gnerated synthetic data
torchrun --nproc_per_node=8 generate_dataset.py --out_dir imagenet_syn/ns750k --num_images 1 --batch_size 1 --cfg 1.25


### fine-tune model
torchrun --nproc_per_node=8 train.py --depth=16 --bs=768 --fp16=1 --alng=1e-4 --data_path imagenet_syn/ns750k --resume var_d16.pth --local_out_dir_path training-runs/ns1k --save_interval 25 --total_images 75 --target_lr 1e-5


#### eval 


# ─────────── Sub-folder (just change this) ───────────
sub_folder="ns750k"

# ─────────── Detect GPUs and set WORLD_SIZE ───────────
WORLD_SIZE=${WORLD_SIZE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# ─────────── Paths & Scripts ───────────
BASE_SCRIPT="generate_and_fid_neon.py"
VAR_CKPT="var_d16.pth"
FID_STATS="fid_stats/adm_in256_stats.npz"
AUX_CKPT_DIR="training-runs/${sub_folder}"

OUT_DIR_FID="results/${sub_folder}"
FINAL_OUT_DIR_FID="results_final/${sub_folder}"
mkdir -p "$OUT_DIR_FID" "$FINAL_OUT_DIR_FID"

# ─────────── Grid Parameters ───────────
EPOCHS="250368 500736 750336 1000704 1250304 1500672 1750272 2000640 2250240 2500608 2750208 3000576 3250176 3500544 3750144 4000512 4250112 4500480 4750080 5000448 5250048 5500416 5750016 6000384 6250752 6500352 6750720 7000320 7250688"
W_VALUES="0.1 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0"
CFG_VALUES="1.35 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0"

# ─────────── Sampling Settings ───────────
NUM_IMAGES=10000
NUM_IMAGES_FINAL=50000
CUDA_FLAG="--cuda"
TOLERANCE=0.03

# ─────────── "Infinite" default for comparisons ───────────
HIGH=1000000

# ─────────── Helper to Read FID from .txt ───────────
get_fid() {
  head -n1 "$1"
}

# ─────────── Epoch Loop ───────────
for epoch in $EPOCHS; do
  AUX_CKPT="${AUX_CKPT_DIR}/ar-ckpt-images_${epoch}.pth"
  echo
  echo "===== VAR evaluation ─ sub_folder=${sub_folder}, epoch=${epoch} ====="

  last_min=$HIGH
  stop_w=false
  best_fid=$HIGH; best_w=0; best_cfg=0

  # 1) Coarse grid sweep over w and cfg
  for w in $W_VALUES; do
    $stop_w && break
    min_fid_w=$HIGH; stop_cfg=false

    for cfg in $CFG_VALUES; do
      $stop_cfg && break

      dir_epoch="${OUT_DIR_FID}/${epoch}"
      mkdir -p "$dir_epoch"
      fid_file="${dir_epoch}/fid_w${w}_cfg${cfg}.txt"

      if [[ -f "$fid_file" ]]; then
        fid=$(get_fid "$fid_file")
        echo "✔ epoch ${epoch} | w=${w}, cfg=${cfg}, Cached FID=${fid}"
      else
        if ! torchrun --standalone --nproc_per_node=$WORLD_SIZE \
            "$BASE_SCRIPT" \
            --var_ckpt "$VAR_CKPT" \
            --var_aux "$AUX_CKPT" \
            --fid_stats "$FID_STATS" \
            --out_dir "$OUT_DIR_FID" \
            $CUDA_FLAG \
            --cfg "$cfg" \
            --w "$w" \
            --N "$NUM_IMAGES"; then
          echo "✗ epoch ${epoch} | w=${w}, cfg=${cfg} → generate_and_fid_neon.py failed, skipping."
          continue
        fi
        fid=$(get_fid "$fid_file")
        echo "✔ epoch ${epoch} | w=${w}, cfg=${cfg}, Computed FID=${fid}"
      fi

      # update per-w min and overall best
      (( $(echo "$fid < $min_fid_w" | bc -l) )) && min_fid_w=$fid
      if (( $(echo "$fid < $best_fid" | bc -l) )); then
        best_fid=$fid; best_w=$w; best_cfg=$cfg
      fi

      # early‐stop cfg loop if performance degrades
      if (( $(echo "$fid > $min_fid_w + $TOLERANCE" | bc -l) )); then
        stop_cfg=true
        echo "*** epoch ${epoch} | w=${w}, Best computed FID=${min_fid_w}"
      fi
    done

    # early‐stop w loop if no improvement
    if (( $(echo "$min_fid_w > $last_min + $TOLERANCE" | bc -l) )); then
      stop_w=true
    fi
    last_min=$min_fid_w
  done

  # 2) Refinement around the best (±0.1)
  echo "Refining around w=${best_w}, cfg=${best_cfg}"
  for dw in -0.1 0 0.1; do
    w_r=$(printf '%.1f' "$(echo "$best_w + $dw" | bc)")
    (( $(echo "$w_r <= 0" | bc -l) )) && continue

    for dc in -0.1 0 0.1; do
      cfg_r=$(printf '%.1f' "$(echo "$best_cfg + $dc" | bc)")
      (( $(echo "$cfg_r <= 0" | bc -l) )) && continue
      [[ "$dw" == 0 && "$dc" == 0 ]] && continue

      dir_epoch="${OUT_DIR_FID}/${epoch}"
      mkdir -p "$dir_epoch"
      fid_file="${dir_epoch}/fid_w${w_r}_cfg${cfg_r}.txt"

      if [[ -f "$fid_file" ]]; then
        fid=$(get_fid "$fid_file")
        echo "✔ epoch ${epoch} | w=${w_r}, cfg=${cfg_r}, Cached FID=${fid}"
      else
        if ! torchrun --standalone --nproc_per_node=$WORLD_SIZE \
            "$BASE_SCRIPT" \
            --var_ckpt "$VAR_CKPT" \
            --var_aux "$AUX_CKPT" \
            --fid_stats "$FID_STATS" \
            --out_dir "$OUT_DIR_FID" \
            $CUDA_FLAG \
            --cfg "$cfg_r" \
            --w "$w_r" \
            --N "$NUM_IMAGES"; then
          echo "✗ epoch ${epoch} | w=${w_r}, cfg=${cfg_r} → refinement failed, skipping."
          continue
        fi
        fid=$(get_fid "$fid_file")
        echo "✔ epoch ${epoch} | w=${w_r}, cfg=${cfg_r}, Computed FID=${fid}"
      fi

      if (( $(echo "$fid < $best_fid" | bc -l) )); then
        best_fid=$fid; best_w=$w_r; best_cfg=$cfg_r
      fi
    done
  done

  echo "*** Epoch ${epoch} best → w=${best_w}, cfg=${best_cfg}, FID=${best_fid} ***"

  # 3) Final 50 K evaluation at best (w, cfg)
  final_dir="${FINAL_OUT_DIR_FID}/${epoch}"
  mkdir -p "$final_dir"
  final_file="${final_dir}/fid_w${best_w}_cfg${best_cfg}.txt"

  if [[ -f "$final_file" ]]; then
    fid=$(get_fid "$final_file")
    echo "✅ Final 50K exists for epoch ${epoch} → FID=${fid}"
  else
    echo "→ FINAL 50 K run for epoch ${epoch}"
    if ! torchrun --standalone --nproc_per_node=$WORLD_SIZE \
        "$BASE_SCRIPT" \
        --var_ckpt "$VAR_CKPT" \
        --var_aux "$AUX_CKPT" \
        --fid_stats "$FID_STATS" \
        --out_dir "$FINAL_OUT_DIR_FID" \
        $CUDA_FLAG \
        --cfg "$best_cfg" \
        --w "$best_w" \
        --N "$NUM_IMAGES_FINAL"; then
      echo "✗ Final 50K failed for epoch ${epoch}, skipping final evaluation."
    else
      fid=$(get_fid "$final_file")
      echo "✔ Final 50K FID=${fid}"
    fi
  fi
done
