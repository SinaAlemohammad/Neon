


####### |S| = 750k ###########


# ─────────── Generate synthetic data ───────────

torchrun --nproc_per_node=8 generate_dataset.py --out_dir imagenet_syn/ns750k --num_images 750 --batch_size 10 --cfg 2.0 --top_k 900 --top_p 0.95 --cuda

# ─────────── Fine-tune ───────────

torchrun --nproc_per_node=8 train.py --depth=36 --bs=24 --fp16=1 --alng=5e-6 --data_path imagenet_syn/ns750k --resume var_d36.pth --local_out_dir_path training-runs/ns750k --save_interval 200000 --total_images 3001000 --target_lr 1e-6 --saln=1 --pn=512 

# ─────────── Sub-folder ───────────
sub_folder="ns750k"

# ─────────── Environment & Logging ───────────
export OMP_NUM_THREADS=1
export LOGLEVEL=ERROR
export PYTHONWARNINGS=ignore

# ─────────── Detect GPUs and set WORLD_SIZE ───────────
WORLD_SIZE=${WORLD_SIZE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# ─────────── Paths & Scripts ───────────
BASE_SCRIPT="generate_and_fid_neon.py"
VAR_CKPT="var_d36.pth"
FID_STATS="fid_stats/adm_in512_stats.npz"
AUX_CKPT_DIR="training-runs/${sub_folder}"

OUT_DIR_FID="results/${sub_folder}"
FINAL_OUT_DIR_FID="results_final/${sub_folder}"
mkdir -p "$OUT_DIR_FID" "$FINAL_OUT_DIR_FID"

# ─────────── Grid Parameters ───────────
EPOCHS="200016 400008 600000 800016 1000008 1200000 1400016 1600008 1800000 2000016 2200008 2400000 2600016 2800008 3000000"
W_VALUES="0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0"
CFG_VALUES="2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4"

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