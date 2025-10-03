set -euo pipefail
IFS=$'\n\t'

# ---- generate synthetic data and construct the labels ----

torchrun --nproc_per_node=8 generate_images.py --config-name=im256_generate_images.yaml eval.resume=imagenet256_ts_a2.pkl +per_class_count=30 +out_dir=imagenet_syn/ns30k/train

python create_labels.py --input_dir imagenet_syn/ns30k/train/ --output_file imagenet_syn/ns30k/train/dataset.json

# ---- encode into latents and zip it for training ----

python dataset_tool.py encode --source=imagenet_syn/ns30k/train --dest=imagenet_syn/ns30k.zip

# ---- fine-tune ----

bash run_train.sh 8 im256.yaml dataset.path=imagenet_syn/ns30k/train

# ---- Fixed ----
NS="ns30k"
TOL=0.03

# ---- Quiet mode ----
export OMP_NUM_THREADS=1
export PYTHONWARNINGS=ignore
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=ERROR

# ---- Tunables ----
NPROC=8
CFG_FILE="im256_generate_images.yaml"
GEN_SCRIPT="generate_fid_neon.py"
FID_STATS="fid_stats/adm_in256_stats.npz"
PER_CLASS_COUNT_SWEEP=10
PER_CLASS_COUNT_FINAL=50

AUX_DIR="outputs/imagenet/${NS}"
shopt -s nullglob

# --- helpers ---
min_val()   { awk -v a="$1" -v b="$2" 'BEGIN{print (a<b)?a:b}'; }
add_tol()   { awk -v a="$1" -v t="$2" 'BEGIN{printf "%.8f", a+t}'; }
gt()        { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }  # exit 0 if a>b
lt()        { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b)}'; }  # exit 0 if a<b
parse_fid() { awk '{if (tolower($1)=="fid:"){print $2; exit}}' "$1"; }

print_row() {
  printf "w=%-5s | cfg=%-5s | FID=%-12s" "$1" "$2" "$3"
  if [[ "${4-}" != "" ]]; then
    printf " | %s" "$4"
  fi
  printf "\n"
}

for STEPS in 1 2 4 8; do
  case $STEPS in
    1)
      neon_WEIGHTS=(0 0.2 0.4 0.8 1.2 1.6 2 2.4 2.8 3.2 3.6 4 4.4 4.8 5.2)
      CFG_SCALES=(1.3 1.4 1.5 1.6 1.7 1.8 2.2 2.6)
      ;;
    2)
      neon_WEIGHTS=(0.4 0.8 1.2 1.6 2 2.4 2.8 3.2 3.6 4 4.4 4.8 5.2)
      CFG_SCALES=(1.6 1.7 1.8 2.2 2.6)
      ;;
    4|8)
      neon_WEIGHTS=(0.8 1.2 1.6 2 2.4 2.8 3.2 3.6 4 4.4 4.8 5.2)
      CFG_SCALES=(1.6 1.7 1.8 2.2 2.6)
      ;;
    *)
      echo "Unsupported STEPS=${STEPS}" >&2
      exit 1
      ;;
  esac

  PRESET="${STEPS}_steps_cfg1.5_pushforward_uniform"
  RESULTS_ROOT="results_new/${NS}/${STEPS}"
  RESULTS_FINAL_ROOT="result_final_new/${NS}/${STEPS}"
  mkdir -p "${RESULTS_ROOT}" "${RESULTS_FINAL_ROOT}"

  for CKPT in "${AUX_DIR}"/network-snapshot-*.pkl; do
    BASENAME="$(basename "$CKPT")"
    AUX_NUM_PAD="$(echo "$BASENAME" | sed -E 's/.*network-snapshot-([0-9]{6,})\.pkl/\1/')"
    DEST_DIR="${RESULTS_ROOT}/${AUX_NUM_PAD}"
    FINAL_DIR="${RESULTS_FINAL_ROOT}/${AUX_NUM_PAD}"
    mkdir -p "${DEST_DIR}" "${FINAL_DIR}"

    echo "============ model and step specifications ========="
    echo "Model checkpoint: ${BASENAME} | Step count: ${STEPS}"
    echo

    # Track best single FID across all (w,cfg) for final re-run
    best_overall_fid="1000000000"
    best_overall_w=""
    best_overall_cfg=""

    # Track best per-W min seen so far (for W-level break rule #2)
    best_wmin_so_far="1000000000"

    for W in "${neon_WEIGHTS[@]}"; do
      cur_w_min="1000000000"              # min FID encountered for THIS W over CFGs seen so far

      for CFG in "${CFG_SCALES[@]}"; do
        DEST_TXT="${DEST_DIR}/w${W}_cfg${CFG}.txt"
        FID_VAL=""
        cache_tag=""

        if [[ -f "${DEST_TXT}" ]]; then
          FID_VAL="$(parse_fid "${DEST_TXT}")"
          cache_tag="cache"
        else
          OUT_TMP="$(mktemp -d -t gen_${NS}_${STEPS}_${AUX_NUM_PAD}_w${W}_cfg${CFG}_XXXX)"
          torchrun --nproc_per_node="${NPROC}" "${GEN_SCRIPT}" \
            --config-name="${CFG_FILE}" \
            +aux_resume="${CKPT}" \
            +neon_w="${W}" \
            +per_class_count="${PER_CLASS_COUNT_SWEEP}" \
            +cfg_scale="${CFG}" \
            +preset="${PRESET}" \
            +out_dir="${OUT_TMP}" \
            +fid_stats="${FID_STATS}" \
            +fid_out="${DEST_TXT}" > /dev/null
          rm -rf "${OUT_TMP}"
          FID_VAL="$(parse_fid "${DEST_TXT}")"
        fi

        [[ -n "${FID_VAL}" ]] || continue

        fid_print="${FID_VAL}"; [[ -n "${cache_tag}" ]] && fid_print="${FID_VAL} ${cache_tag}"

        # ---- Rule #1: CFG-level early stop vs min-so-far FOR THIS W ----
        THRESH_CFG="$(add_tol "${cur_w_min}" "${TOL}")"
        if gt "${FID_VAL}" "${THRESH_CFG}"; then
          print_row "${W}" "${CFG}" "${fid_print}" "break cfg: FID=${FID_VAL} > minW_so_far=${cur_w_min}+${TOL}"
          break
        fi

        # Update per-W minimum and global best (for final selection)
        cur_w_min="$(min_val "${cur_w_min}" "${FID_VAL}")"
        if lt "${FID_VAL}" "${best_overall_fid}"; then
          best_overall_fid="${FID_VAL}"
          best_overall_w="${W}"
          best_overall_cfg="${CFG}"
        fi

        print_row "${W}" "${CFG}" "${fid_print}"
      done

      # If we never parsed a valid FID for this W, skip W-level check
      if [[ "${cur_w_min}" == "1000000000" ]]; then
        echo "No valid FIDs for w=${W}; skipping W-level check."
        continue
      fi

      # ---- Rule #2: W-level early stop vs best W-min seen so far ----
      THRESH_W="$(add_tol "${best_wmin_so_far}" "${TOL}")"
      if gt "${cur_w_min}" "${THRESH_W}"; then
        print_row "${W}" "--" "${cur_w_min}" "break w: minFID(W)=${cur_w_min} > bestWmin_so_far=${best_wmin_so_far}+${TOL}"
        break
      fi

      # Update global best W-min baseline
      best_wmin_so_far="$(min_val "${best_wmin_so_far}" "${cur_w_min}")"
    done

    # ---- Final evaluation for the best (w,cfg) found ----
    if [[ -n "${best_overall_w}" && -n "${best_overall_cfg}" ]]; then
      FINAL_TXT="${FINAL_DIR}/w${best_overall_w}_cfg${best_overall_cfg}.txt"
      if [[ -f "${FINAL_TXT}" ]]; then
        echo "final: exists w=${best_overall_w} cfg=${best_overall_cfg}"
      else
        OUT_TMP="$(mktemp -d -t final_${NS}_${STEPS}_${AUX_NUM_PAD}_w${best_overall_w}_cfg${best_overall_cfg}_XXXX)"
        echo "final: w=${best_overall_w} cfg=${best_overall_cfg}"
        torchrun --nproc_per_node="${NPROC}" "${GEN_SCRIPT}" \
          --config-name="${CFG_FILE}" \
          +aux_resume="${CKPT}" \
          +neon_w="${best_overall_w}" \
          +per_class_count="${PER_CLASS_COUNT_FINAL}" \
          +cfg_scale="${best_overall_cfg}" \
          +preset="${PRESET}" \
          +out_dir="${OUT_TMP}" \
          +fid_stats="${FID_STATS}" \
          +fid_out="${FINAL_TXT}" > /dev/null
        rm -rf "${OUT_TMP}"
      fi
      FID_FINAL="$(parse_fid "${FINAL_TXT}")"
      echo "Final best (per_class_count=${PER_CLASS_COUNT_FINAL}):"
      print_row "${best_overall_w}" "${best_overall_cfg}" "${FID_FINAL}"
    else
      echo "final: none"
    fi

    echo
  done
done

