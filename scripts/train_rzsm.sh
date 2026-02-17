#!/usr/bin/env bash
# Train RZSM (our method), evaluate all methods, and plot.
#
# Requires vanilla checkpoints to already exist in logs/<env>/vanilla/.
# Train baselines first:  bash scripts/train_baselines.sh --env <ENV>
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

echo "=========================================="
echo "  Train RZSM"
echo "  Env:       $ENV"
echo "  Steps:     $STEPS  (${EPOCHS} epochs x ${STEPS_PER_EPOCH})"
echo "  Seeds:     $SEEDS"
echo "  Num envs:  $NUM_ENVS"
echo "  Device:    $DEVICE"
echo "  Vanilla:   $VANILLA_DIR"
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
    --disturbance-ratio 0.05 --disturbance-prob 0.3 \
    --pi-opt-path "$PI_OPT"
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
echo ""
echo ">>> RZSM training complete in ${TRAIN_ELAPSED}s"

# ── Evaluation: all 5 methods ────────────────────────────────────────
echo ""
METHODS="vanilla,rarl,sa_mdp,dr,rzsm"
echo ">>> Running evaluation on all methods ($EVAL_EPS episodes per seed)..."
python -m src.eval \
  --env "$ENV" \
  --methods "$METHODS" \
  --episodes "$EVAL_EPS" \
  --checkpoint-dir "$LOG_BASE" \
  --device "$DEVICE" \
  --output results/

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
echo "  Checkpoints: ${LOG_BASE}/rzsm/seed_*/"
echo "  Results:     results/${ENV_SAFE}/"
echo "=========================================="
