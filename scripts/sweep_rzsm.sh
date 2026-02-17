#!/usr/bin/env bash
# Hyperparameter sweep for RZSM on a given environment.
#
# Tests combinations of key hyperparameters and saves results for comparison.
# Uses 1 seed and shorter training (200K steps) for fast iteration.
# After finding the best config, run full training with train_rzsm.sh.
#
# Usage:
#   bash scripts/sweep_rzsm.sh --env Ant-v5
#   bash scripts/sweep_rzsm.sh --env Ant-v5 --steps 500000 --num-envs 50
#   bash scripts/sweep_rzsm.sh --env HalfCheetah-v5 --seeds 2

set -euo pipefail

# ── Activate virtual environment ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_DIR/.env/bin/activate" ]]; then
  source "$PROJECT_DIR/.env/bin/activate"
fi
cd "$PROJECT_DIR"

# ── Defaults ─────────────────────────────────────────────────────────
ENV=""
STEPS=200000
SEEDS=1
NUM_ENVS=100
DEVICE="cuda"
EVAL_EPS=5

# ── Parse named arguments ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)        ENV="$2";        shift 2 ;;
    --steps)      STEPS="$2";      shift 2 ;;
    --seeds)      SEEDS="$2";      shift 2 ;;
    --num-envs)   NUM_ENVS="$2";   shift 2 ;;
    --device)     DEVICE="$2";     shift 2 ;;
    --eval-eps)   EVAL_EPS="$2";   shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/sweep_rzsm.sh --env <ENV> [OPTIONS]"
      echo ""
      echo "Hyperparameter sweep for RZSM. Tests multiple configurations."
      echo ""
      echo "Options:"
      echo "  --env        Gymnasium environment ID       (required)"
      echo "  --steps      Steps per config               (default: 200000)"
      echo "  --seeds      Seeds per config               (default: 1)"
      echo "  --num-envs   Parallel vectorized envs       (default: 100)"
      echo "  --device     Torch device                   (default: cuda)"
      echo "  --eval-eps   Eval episodes per config       (default: 5)"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$ENV" ]]; then
  echo "ERROR: --env is required."
  exit 1
fi

# ── Compute steps_per_epoch and epochs ────────────────────────────────
STEPS_PER_EPOCH=$(( NUM_ENVS * 1000 ))
if (( STEPS_PER_EPOCH < 4000 )); then
  STEPS_PER_EPOCH=4000
fi
EPOCHS=$(( STEPS / STEPS_PER_EPOCH ))
if (( EPOCHS < 1 )); then
  EPOCHS=1
fi
START_STEPS=25000
UPDATE_AFTER=25000

ENV_SAFE="${ENV//-/_}"
SWEEP_DIR="logs/${ENV_SAFE}/sweep_rzsm"
VANILLA_DIR="logs/${ENV_SAFE}/vanilla"
RESULTS_CSV="${SWEEP_DIR}/sweep_results.csv"

# ── Check vanilla checkpoints ────────────────────────────────────────
if [[ ! -d "$VANILLA_DIR" ]]; then
  echo "ERROR: Vanilla checkpoints not found at: $VANILLA_DIR"
  echo "  Train baselines first: bash scripts/train_baselines.sh --env $ENV"
  exit 1
fi

# ── Define sweep configurations ──────────────────────────────────────
# Format: "name|hidden_sizes|seq_len|d_model|nhead|layers|det_lr|det_interval|dist_ratio|dist_prob"
#
# Sweep axes:
#   1. Network size:        small(256,256) vs large(400,300)
#   2. Transformer size:    small(d64,L2) vs med(d128,L3) vs large(d256,L4)
#   3. Sequence length:     10, 20, 40
#   4. Disturbance ratio:   0.03, 0.05, 0.10
#   5. Detector LR:         1e-4, 3e-4
#   6. Detector interval:   100, 200

CONFIGS=(
  # ── Baseline (current defaults) ──
  "baseline|256,256|20|128|4|3|1e-4|200|0.05|0.3"

  # ── Network size ──
  "net_large|400,300|20|128|4|3|1e-4|200|0.05|0.3"

  # ── Transformer size ──
  "tf_small|256,256|20|64|4|2|1e-4|200|0.05|0.3"
  "tf_large|256,256|20|256|4|4|1e-4|200|0.05|0.3"

  # ── Sequence length ──
  "seq10|256,256|10|128|4|3|1e-4|200|0.05|0.3"
  "seq40|256,256|40|128|4|3|1e-4|200|0.05|0.3"

  # ── Disturbance ratio ──
  "dist003|256,256|20|128|4|3|1e-4|200|0.03|0.3"
  "dist010|256,256|20|128|4|3|1e-4|200|0.10|0.3"

  # ── Detector learning rate ──
  "detlr3e4|256,256|20|128|4|3|3e-4|200|0.05|0.3"

  # ── Detector training interval ──
  "detint100|256,256|20|128|4|3|1e-4|100|0.05|0.3"

  # ── Combined: large net + large transformer ──
  "big_both|400,300|30|256|4|4|1e-4|100|0.05|0.3"

  # ── Combined: weaker adversary + bigger detector ──
  "weak_adv_big_det|256,256|30|256|4|4|1e-4|100|0.03|0.2"

  # ── Combined: large net + weaker adversary ──
  "net_large_weak|400,300|20|128|4|3|1e-4|200|0.03|0.2"
)

TOTAL=${#CONFIGS[@]}

mkdir -p "$SWEEP_DIR"

echo "=========================================="
echo "  RZSM Hyperparameter Sweep"
echo "  Env:       $ENV"
echo "  Steps:     $STEPS  (${EPOCHS} epochs x ${STEPS_PER_EPOCH})"
echo "  Seeds:     $SEEDS"
echo "  Configs:   $TOTAL"
echo "  Num envs:  $NUM_ENVS"
echo "  Device:    $DEVICE"
echo "  Output:    $SWEEP_DIR"
echo "=========================================="

# ── Write CSV header ─────────────────────────────────────────────────
echo "config,hidden_sizes,seq_len,d_model,nhead,layers,det_lr,det_interval,dist_ratio,dist_prob,seed,final_return,mean_return_last3,eval_nominal,eval_force,eval_combined" \
  > "$RESULTS_CSV"

SWEEP_START=$(date +%s)
COMPLETED=0

for cfg_str in "${CONFIGS[@]}"; do
  IFS='|' read -r CFG_NAME HIDDEN SEQ DMODEL NHEAD LAYERS DET_LR DET_INT DIST_R DIST_P <<< "$cfg_str"
  COMPLETED=$((COMPLETED + 1))

  echo ""
  echo "=========================================="
  echo "  [$COMPLETED/$TOTAL] Config: $CFG_NAME"
  echo "  hidden=$HIDDEN  seq=$SEQ  d_model=$DMODEL  nhead=$NHEAD  layers=$LAYERS"
  echo "  det_lr=$DET_LR  det_interval=$DET_INT  dist_ratio=$DIST_R  dist_prob=$DIST_P"
  echo "=========================================="

  CFG_DIR="${SWEEP_DIR}/${CFG_NAME}"

  # Clean previous run of same config
  if [[ -d "$CFG_DIR" ]]; then
    rm -rf "$CFG_DIR"
  fi

  for seed in $(seq 0 $((SEEDS - 1))); do
    PI_OPT="${VANILLA_DIR}/seed_${seed}/checkpoints"
    if [[ ! -d "$PI_OPT" ]]; then
      echo "  [WARN] No vanilla seed_${seed}, skipping."
      continue
    fi

    echo "  Training seed=$seed..."
    python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
      --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
      --start-steps "$START_STEPS" --update-after "$UPDATE_AFTER" \
      --num-envs "$NUM_ENVS" --device "$DEVICE" \
      --log-dir "${CFG_DIR}/rzsm" \
      --hidden-sizes "$HIDDEN" \
      --seq-len "$SEQ" --d-model "$DMODEL" --nhead "$NHEAD" \
      --transformer-layers "$LAYERS" --detector-lr "$DET_LR" \
      --detector-train-interval "$DET_INT" \
      --disturbance-ratio "$DIST_R" --disturbance-prob "$DIST_P" \
      --pi-opt-path "$PI_OPT"

    # ── Extract training returns ──────────────────────────────────
    RETURNS_CSV="${CFG_DIR}/rzsm/seed_${seed}/training_returns.csv"
    FINAL_RET="nan"
    MEAN_LAST3="nan"
    if [[ -f "$RETURNS_CSV" ]]; then
      FINAL_RET=$(tail -1 "$RETURNS_CSV" | cut -d',' -f2)
      MEAN_LAST3=$(tail -3 "$RETURNS_CSV" | cut -d',' -f2 | \
        python3 -c "import sys; vals=[float(x) for x in sys.stdin if x.strip()]; print(f'{sum(vals)/len(vals):.2f}' if vals else 'nan')")
    fi

    # ── Quick eval: nominal, force, combined ──────────────────────
    EVAL_NOM="nan"
    EVAL_FORCE="nan"
    EVAL_COMB="nan"

    echo "  Evaluating seed=$seed..."
    # We need to copy vanilla to cfg_dir so eval can find it
    if [[ ! -d "${CFG_DIR}/vanilla" ]]; then
      ln -sf "$(cd "$VANILLA_DIR" && pwd)" "${CFG_DIR}/vanilla"
    fi

    EVAL_OUT=$(python -m src.eval \
      --env "$ENV" \
      --methods rzsm \
      --scenarios nominal,force,combined \
      --episodes "$EVAL_EPS" \
      --checkpoint-dir "$CFG_DIR" \
      --device "$DEVICE" \
      --output "${CFG_DIR}/results" \
      --seq-len "$SEQ" --d-model "$DMODEL" --transformer-layers "$LAYERS" \
      --hidden-sizes "$HIDDEN" \
      2>&1) || true

    # Parse eval output for returns
    EVAL_NOM=$(echo "$EVAL_OUT" | grep "rzsm.*nominal" | grep -oP '[\d.]+±' | head -1 | tr -d '±' || echo "nan")
    EVAL_FORCE=$(echo "$EVAL_OUT" | grep "rzsm.*force" | grep -oP '[\d.]+±' | head -1 | tr -d '±' || echo "nan")
    EVAL_COMB=$(echo "$EVAL_OUT" | grep "rzsm.*combined" | grep -oP '[\d.]+±' | head -1 | tr -d '±' || echo "nan")

    # If grep didn't work, try extracting from CSV
    EVAL_CSV="${CFG_DIR}/results/${ENV_SAFE}/comparison_table.csv"
    if [[ -f "$EVAL_CSV" ]]; then
      EVAL_NOM=$(grep "rzsm,nominal" "$EVAL_CSV" | cut -d',' -f4 || echo "nan")
      EVAL_FORCE=$(grep "rzsm,force" "$EVAL_CSV" | cut -d',' -f4 || echo "nan")
      EVAL_COMB=$(grep "rzsm,combined" "$EVAL_CSV" | cut -d',' -f4 || echo "nan")
    fi

    echo "$CFG_NAME,\"$HIDDEN\",$SEQ,$DMODEL,$NHEAD,$LAYERS,$DET_LR,$DET_INT,$DIST_R,$DIST_P,$seed,$FINAL_RET,$MEAN_LAST3,$EVAL_NOM,$EVAL_FORCE,$EVAL_COMB" \
      >> "$RESULTS_CSV"

    echo "  seed=$seed: train_return=$FINAL_RET  eval_nom=$EVAL_NOM  eval_force=$EVAL_FORCE  eval_comb=$EVAL_COMB"
  done
done

SWEEP_END=$(date +%s)
SWEEP_ELAPSED=$(( SWEEP_END - SWEEP_START ))

# ── Print summary ────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Sweep Complete!  (${SWEEP_ELAPSED}s)"
echo "  Results: $RESULTS_CSV"
echo "=========================================="
echo ""
echo "  Results sorted by eval_nominal (descending):"
echo ""
echo "  CONFIG                  NOMINAL    FORCE      COMBINED   TRAIN_RET"
echo "  ─────────────────────── ────────── ────────── ────────── ──────────"

# Sort by eval_nominal (column 14) descending, skip header
tail -n +2 "$RESULTS_CSV" | sort -t',' -k14 -rn | while IFS=',' read -r cfg _ _ _ _ _ _ _ _ _ _ fret _ enom eforce ecomb; do
  printf "  %-25s %-10s %-10s %-10s %-10s\n" "$cfg" "$enom" "$eforce" "$ecomb" "$fret"
done

echo ""
echo "  To train with the best config, update train_rzsm.sh or run:"
echo "    python -m src.train --env $ENV --mode adversarial \\"
echo "      --hidden-sizes <best> --seq-len <best> --d-model <best> \\"
echo "      --transformer-layers <best> --detector-lr <best> \\"
echo "      --disturbance-ratio <best> --disturbance-prob <best> ..."
echo ""
