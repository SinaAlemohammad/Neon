
#!/usr/bin/env bash


torchrun --standalone --nproc_per_node=8 generate_dataset.py --model xar_large --model_ckpt pretrained/xAR-L.pth --vae_path pretrained/kl16.ckpt --num_images 30 --batch_size 64 --flow_steps 50 --cfg 2.3 --img_size 256 --out_dir imagenet_syn/ns30k/train

torchrun --nproc_per_node=8 train_new.py --img_size 256 --vae_path pretrained/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model xar_large --epochs 10000 --warmup_epochs 10000 --batch_size 32 --blr 1e-6 --output_dir training-runs/ns30k --resume pretrained/xAR-L.pth --data_path imagenet_syn/ns30k



set -euo pipefail

# ─────────── Configuration ───────────
export OMP_NUM_THREADS=1
export LOGLEVEL=ERROR
export PYTHONWARNINGS=ignore

BASE_MODEL_SCRIPT="generate_and_fid_sims.py"
MODEL_VARIANT="xar_large"
BASE_CKPT="pretrained/xAR-L.pth"
VAE_PATH="pretrained/kl16.ckpt"
AUX_CKPT_DIR="training-runs/ns30k"

OUT_DIR_IMAGES="temp_images"
OUT_DIR_FID="results/30k"
FINAL_OUT_DIR_FID="results_final/ns30k"

NUM_IMAGES=10000
NUM_IMAGES_FINAL=50000
BATCH_SIZE=64
FLOW_STEPS=50
IMG_SIZE=256
CUDA_FLAG="--cuda"
WORLD_SIZE=8
TOLERANCE=0.02

EPOCHS="250112 500224 750080 1000192 1250048 1500160 1750016 2000128 2250240 2500096 2750208 3000064 3250176 3500032 3750144 4000000 4250112 4500224 4750080 5000192 5250048 5500160 5750016 6000128 6250240 6500096 6750208 7000064 7250176"
W_VALUES="0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8"
CFG_VALUES="2.4 2.6 2.8 3.0 3.2 3.4 3.6"

mkdir -p "$OUT_DIR_FID" "$FINAL_OUT_DIR_FID"

# helper — grab fid from json
get_fid () { python3 -c "import json,sys; print(json.load(open('$1'))['fid'])"; }

# ─────────── Epoch loop ───────────
for epoch in $EPOCHS; do
  AUX_CKPT="$AUX_CKPT_DIR/checkpoint-checkpoint-${epoch}.pth"

  echo -e "\n==============================="
  echo    "       Epoch  $epoch"
  echo    "==============================="

  last_min_fid=1000
  stop_w_loop=false
  global_min_fid=1000; global_min_w=0; global_min_cfg=0

  # ❶ coarse grid
  for w in $W_VALUES; do
    $stop_w_loop && break
    echo "--- w = $w ---"

    min_fid_this_w=1000
    stop_cfg_loop=false

    for cfg in $CFG_VALUES; do
      $stop_cfg_loop && break

      epoch_dir="$OUT_DIR_FID/$epoch"
      mkdir -p "$epoch_dir"
      json_file="$epoch_dir/cfg=$(printf '%.3f' $cfg)_w=$(printf '%.3f' $w).json"

      if [[ -f "$json_file" ]]; then
        echo "⏩  Found JSON for w=$w cfg=$cfg"
        fid=$(get_fid "$json_file")
      else
        echo "▶  Run ep${epoch} w=$w cfg=$cfg"
        torchrun --standalone --nproc_per_node=$WORLD_SIZE \
          "$BASE_MODEL_SCRIPT" \
          --model "$MODEL_VARIANT" \
          --model_ckpt "$BASE_CKPT" \
          --model_ckpt_aux "$AUX_CKPT" \
          --w "$w" \
          --cfg "$cfg" \
          --vae_path "$VAE_PATH" \
          --num_images "$NUM_IMAGES" \
          --batch_size "$BATCH_SIZE" \
          --flow_steps "$FLOW_STEPS" \
          --img_size "$IMG_SIZE" \
          $CUDA_FLAG \
          --out_dir "$OUT_DIR_IMAGES" \
          --out_dir_fid "$OUT_DIR_FID"
        fid=$(get_fid "$json_file")
      fi

      # minima tracking
      (( $(echo "$fid < $min_fid_this_w" | bc -l) )) && min_fid_this_w=$fid
      if (( $(echo "$fid < $global_min_fid" | bc -l) )); then
        global_min_fid=$fid; global_min_w=$w; global_min_cfg=$cfg
      fi

      # cfg early‑stop
      if (( $(echo "$fid > $min_fid_this_w + $TOLERANCE" | bc -l) )); then
        echo "↑ FID worsened → stop cfg loop at w=$w"
        stop_cfg_loop=true
      fi
    done

    # w early‑stop
    if (( $(echo "$min_fid_this_w > $last_min_fid + $TOLERANCE" | bc -l) )); then
      echo "↑ No w‑improvement → stop w sweep"
      stop_w_loop=true
    fi
    last_min_fid=$min_fid_this_w
  done  # coarse grid

  # ❷ refinement around *frozen* centre
  centre_w=$global_min_w
  centre_cfg=$global_min_cfg
  echo -e "\n*** Refinement around centre (w=$centre_w, cfg=$centre_cfg) ***"

  for dw in -0.1 0 0.1; do
    w_ref=$(printf "%.3f" "$(echo "$centre_w + $dw" | bc)")
    (( $(echo "$w_ref <= 0" | bc -l) )) && continue
    for dc in -0.1 0 0.1; do
      cfg_ref=$(printf "%.3f" "$(echo "$centre_cfg + $dc" | bc)")
      (( $(echo "$cfg_ref <= 0" | bc -l) )) && continue
      [[ "$dw" = 0 && "$dc" = 0 ]] && continue

      epoch_dir="$OUT_DIR_FID/$epoch"
      json_file="$epoch_dir/cfg=$(printf '%.3f' $cfg_ref)_w=$(printf '%.3f' $w_ref).json"

      if [[ -f "$json_file" ]]; then
        echo "⏩  Have refinement w=$w_ref cfg=$cfg_ref"
        fid=$(get_fid "$json_file")
      else
        echo "▶  Refinement w=$w_ref cfg=$cfg_ref"
        torchrun --standalone --nproc_per_node=$WORLD_SIZE \
          "$BASE_MODEL_SCRIPT" \
          --model "$MODEL_VARIANT" \
          --model_ckpt "$BASE_CKPT" \
          --model_ckpt_aux "$AUX_CKPT" \
          --w "$w_ref" \
          --cfg "$cfg_ref" \
          --vae_path "$VAE_PATH" \
          --num_images "$NUM_IMAGES" \
          --batch_size "$BATCH_SIZE" \
          --flow_steps "$FLOW_STEPS" \
          --img_size "$IMG_SIZE" \
          $CUDA_FLAG \
          --out_dir "$OUT_DIR_IMAGES" \
          --out_dir_fid "$OUT_DIR_FID"
        fid=$(get_fid "$json_file")
      fi

      # update global best (centre remains frozen)
      if (( $(echo "$fid < $global_min_fid" | bc -l) )); then
        global_min_fid=$fid; global_min_w=$w_ref; global_min_cfg=$cfg_ref
        echo "★ New epoch‑best → w=$global_min_w cfg=$global_min_cfg FID=$global_min_fid"
      fi
    done
  done

  echo -e "\n*** FINAL epoch‑best: w=${global_min_w}, cfg=${global_min_cfg}, FID=${global_min_fid} ***"

  # ❸ final 50 K run
  final_epoch_dir="$FINAL_OUT_DIR_FID/$epoch"
  mkdir -p "$final_epoch_dir"
  final_json="$final_epoch_dir/cfg=$(printf '%.3f' $global_min_cfg)_w=$(printf '%.3f' $global_min_w).json"

  if [[ -f "$final_json" ]]; then
    echo "✅  50 K JSON already exists → skipping"
  else
    echo "🚀  FINAL 50 K run: w=${global_min_w} cfg=${global_min_cfg}"
    torchrun --standalone --nproc_per_node=$WORLD_SIZE \
      "$BASE_MODEL_SCRIPT" \
      --model "$MODEL_VARIANT" \
      --model_ckpt "$BASE_CKPT" \
      --model_ckpt_aux "$AUX_CKPT" \
      --w "$global_min_w" \
      --cfg "$global_min_cfg" \
      --vae_path "$VAE_PATH" \
      --num_images "$NUM_IMAGES_FINAL" \
      --batch_size "$BATCH_SIZE" \
      --flow_steps "$FLOW_STEPS" \
      --img_size "$IMG_SIZE" \
      $CUDA_FLAG \
      --out_dir "$OUT_DIR_IMAGES" \
      --out_dir_fid "$FINAL_OUT_DIR_FID"
  fi
done  # epoch loop
