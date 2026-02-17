#!/usr/bin/env bash
# Train RZSM (our method), evaluate all methods, and plot.
#
# Automatically loads best hyperparameters from sweep results if available:
#   logs/<env>/sweep_rzsm/sweep_results.csv
# If no sweep results exist, uses default hyperparameters.
#
# Recommended workflow:
#   1. bash scripts/train_baselines.sh --env <ENV>     # train baselines first
#   2. bash scripts/sweep_rzsm.sh --env <ENV>          # find best hyperparams
#   3. bash scripts/train_rzsm.sh --env <ENV>          # auto-uses best config
#
# Requires vanilla checkpoints to already exist in logs/<env>/vanilla/.
#
# Saves checkpoints to: logs/<env>/rzsm/seed_*/
#
# Usage:
#   bash scripts/train_rzsm.sh --env HalfCheetah-v5
#   bash scripts/train_rzsm.sh --env Ant-v5 --steps 2000000 --seeds 3
#   bash scripts/train_rzsm.sh --env Walker2d-v5 --device cpu --num-envs 4

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
STEPS=1000000
SEEDS=5
NUM_ENVS=100
DEVICE="cuda"
EVAL_EPS=10

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
      echo "Usage: bash scripts/train_rzsm.sh --env <ENV> [OPTIONS]"
      echo ""
      echo "Train RZSM (requires vanilla checkpoints from train_baselines.sh)."
      echo ""
      echo "Options:"
      echo "  --env        Gymnasium environment ID       (required)"
      echo "  --steps      Total training steps           (default: 1000000)"
      echo "  --seeds      Number of random seeds         (default: 5)"
      echo "  --num-envs   Parallel vectorized envs       (default: 100)"
      echo "  --device     Torch device                   (default: cuda)"
      echo "  --eval-eps   Eval episodes per seed         (default: 10)"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      echo "Run with --help for usage."
      exit 1
      ;;
  esac
done

if [[ -z "$ENV" ]]; then
  echo "ERROR: --env is required."
  echo "Usage: bash scripts/train_rzsm.sh --env HalfCheetah-v5"
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
LOG_BASE="logs/${ENV_SAFE}"

# ── Check that vanilla checkpoints exist ─────────────────────────────
VANILLA_DIR="${LOG_BASE}/vanilla"
if [[ ! -d "$VANILLA_DIR" ]]; then
  echo "ERROR: Vanilla checkpoints not found at: $VANILLA_DIR"
  echo "  Train baselines first:  bash scripts/train_baselines.sh --env $ENV"
  exit 1
fi

# ── Load best hyperparams from sweep (or use defaults) ───────────────
SWEEP_CSV="${LOG_BASE}/sweep_rzsm/sweep_results.csv"

# Defaults (same as sweep "baseline" config)
HIDDEN_SIZES="256,256"
SEQ_LEN=20
D_MODEL=128
NHEAD=4
TF_LAYERS=3
DET_LR="1e-4"
DET_INTERVAL=200
DIST_RATIO=0.05
DIST_PROB=0.3
SWEEP_CONFIG="(defaults)"

if [[ -f "$SWEEP_CSV" ]]; then
  echo ""
  echo ">>> Found sweep results: $SWEEP_CSV"
  echo ">>> Selecting best config by eval_nominal..."

  # Python one-liner: read CSV, find row with highest eval_nominal, print pipe-separated params
  BEST_LINE=$(python3 -c "
import csv, sys
best, best_row = -float('inf'), None
with open('$SWEEP_CSV') as f:
    for row in csv.DictReader(f):
        try:
            val = float(row['eval_nominal'])
        except (ValueError, KeyError):
            continue
        if val > best:
            best, best_row = val, row
if best_row is None:
    sys.exit(1)
print('|'.join([
    best_row['config'],
    best_row['hidden_sizes'],
    best_row['seq_len'],
    best_row['d_model'],
    best_row['nhead'],
    best_row['layers'],
    best_row['det_lr'],
    best_row['det_interval'],
    best_row['dist_ratio'],
    best_row['dist_prob'],
    str(best),
]))
" 2>/dev/null) || true

  if [[ -n "$BEST_LINE" ]]; then
    IFS='|' read -r SWEEP_CONFIG HIDDEN_SIZES SEQ_LEN D_MODEL NHEAD TF_LAYERS \
                     DET_LR DET_INTERVAL DIST_RATIO DIST_PROB BEST_EVAL <<< "$BEST_LINE"
    echo "  Best config:   $SWEEP_CONFIG  (eval_nominal=$BEST_EVAL)"
    echo "  hidden=$HIDDEN_SIZES  seq=$SEQ_LEN  d_model=$D_MODEL  nhead=$NHEAD  layers=$TF_LAYERS"
    echo "  det_lr=$DET_LR  det_interval=$DET_INTERVAL  dist_ratio=$DIST_RATIO  dist_prob=$DIST_PROB"
  else
    echo "  [WARN] Could not parse sweep results — using defaults."
    SWEEP_CONFIG="(defaults)"
  fi
else
  echo ""
  echo ">>> No sweep results found at: $SWEEP_CSV"
  echo ">>> Using default hyperparameters."
  echo ">>> Tip: run sweep first:  bash scripts/sweep_rzsm.sh --env $ENV"
fi

echo ""
echo "=========================================="
echo "  Train RZSM"
echo "  Env:       $ENV"
echo "  Steps:     $STEPS  (${EPOCHS} epochs x ${STEPS_PER_EPOCH})"
echo "  Seeds:     $SEEDS"
echo "  Num envs:  $NUM_ENVS"
echo "  Device:    $DEVICE"
echo "  Vanilla:   $VANILLA_DIR"
echo "  Config:    $SWEEP_CONFIG"
echo "  hidden=$HIDDEN_SIZES  seq=$SEQ_LEN  d_model=$D_MODEL"
echo "  nhead=$NHEAD  layers=$TF_LAYERS  det_lr=$DET_LR  det_int=$DET_INTERVAL"
echo "  dist_ratio=$DIST_RATIO  dist_prob=$DIST_PROB"
echo "  Log dir:   $LOG_BASE/rzsm"
echo "=========================================="

TRAIN_START=$(date +%s)

# ── Delete old RZSM before training ──────────────────────────────────
if [[ -d "${LOG_BASE}/rzsm" ]]; then
  echo ""
  echo ">>> Cleaning old RZSM checkpoints..."
  rm -rf "${LOG_BASE}/rzsm"
  echo "  Removed ${LOG_BASE}/rzsm"
fi

# ── Train RZSM ───────────────────────────────────────────────────────
echo ""
echo ">>> Training RZSM policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  PI_OPT="${VANILLA_DIR}/seed_${seed}/checkpoints"
  if [[ ! -d "$PI_OPT" ]]; then
    echo "  [WARN] No vanilla seed_${seed} checkpoint, skipping."
    continue
  fi
  echo "  [rzsm] seed=$seed  (pi_opt from vanilla/seed_${seed})"
  python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
    --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
    --start-steps "$START_STEPS" --update-after "$UPDATE_AFTER" \
    --num-envs "$NUM_ENVS" --device "$DEVICE" \
    --log-dir "${LOG_BASE}/rzsm" \
    --hidden-sizes "$HIDDEN_SIZES" \
    --seq-len "$SEQ_LEN" --d-model "$D_MODEL" --nhead "$NHEAD" \
    --transformer-layers "$TF_LAYERS" --detector-lr "$DET_LR" \
    --detector-train-interval "$DET_INTERVAL" \
    --disturbance-ratio "$DIST_RATIO" --disturbance-prob "$DIST_PROB" \
    --pi-opt-path "$PI_OPT"
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
echo ""
echo ">>> RZSM training complete in ${TRAIN_ELAPSED}s"

# ── Evaluation: all available methods ────────────────────────────────
echo ""
METHODS="rzsm"
for m in vanilla rarl sa_mdp dr; do
  if [[ -d "${LOG_BASE}/${m}" ]]; then
    METHODS="${METHODS},${m}"
  fi
done
echo ">>> Running evaluation on methods: $METHODS ($EVAL_EPS episodes per seed)..."
python -m src.eval \
  --env "$ENV" \
  --methods "$METHODS" \
  --episodes "$EVAL_EPS" \
  --checkpoint-dir "$LOG_BASE" \
  --device "$DEVICE" \
  --output results/ \
  --hidden-sizes "$HIDDEN_SIZES" \
  --seq-len "$SEQ_LEN" --d-model "$D_MODEL" \
  --transformer-layers "$TF_LAYERS"

# ── Plots ────────────────────────────────────────────────────────────
echo ""
echo ">>> Generating plots..."
python scripts/plot_results.py \
  --env "$ENV" \
  --log-dir logs

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TRAIN_START ))

echo ""
echo "=========================================="
echo "  RZSM done!"
echo "  Training:    ${TRAIN_ELAPSED}s"
echo "  Total:       ${TOTAL_ELAPSED}s"
echo "  Config:      $SWEEP_CONFIG"
echo "  Checkpoints: ${LOG_BASE}/rzsm/seed_*/"
echo "  Results:     results/${ENV_SAFE}/"
echo "=========================================="
