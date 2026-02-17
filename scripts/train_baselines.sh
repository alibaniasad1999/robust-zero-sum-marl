#!/usr/bin/env bash
# Train baseline methods (vanilla, rarl, sa_mdp, dr), evaluate all, and plot.
#
# Saves checkpoints to: logs/<env>/{vanilla,rarl,sa_mdp,dr}/seed_*/
# If RZSM checkpoints exist in logs/<env>/rzsm/, they are included in eval.
#
# Usage:
#   bash scripts/train_baselines.sh --env HalfCheetah-v5
#   bash scripts/train_baselines.sh --env Ant-v5 --steps 2000000 --seeds 3
#   bash scripts/train_baselines.sh --env Walker2d-v5 --device cpu --num-envs 4

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
      echo "Usage: bash scripts/train_baselines.sh --env <ENV> [OPTIONS]"
      echo ""
      echo "Train baseline methods (vanilla, rarl, sa_mdp, dr), eval, and plot."
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
  echo "Usage: bash scripts/train_baselines.sh --env HalfCheetah-v5"
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
mkdir -p "$LOG_BASE"

echo "=========================================="
echo "  Train Baselines"
echo "  Env:       $ENV"
echo "  Steps:     $STEPS  (${EPOCHS} epochs x ${STEPS_PER_EPOCH})"
echo "  Seeds:     $SEEDS"
echo "  Num envs:  $NUM_ENVS"
echo "  Device:    $DEVICE"
echo "  Log dir:   $LOG_BASE"
echo "=========================================="

TRAIN_START=$(date +%s)

# ── Helper: check if a method already has all seeds trained ──────────
method_already_trained() {
  local method_dir="$1"
  local needed_seeds="$2"
  if [[ ! -d "$method_dir" ]]; then
    return 1  # not trained
  fi
  for seed in $(seq 0 $((needed_seeds - 1))); do
    if [[ ! -d "${method_dir}/seed_${seed}/checkpoints" ]]; then
      return 1  # missing seed
    fi
  done
  return 0  # all seeds present
}

# ── Phase 1: Vanilla (nominal) ────────────────────────────────────────
echo ""
if method_already_trained "${LOG_BASE}/vanilla" "$SEEDS"; then
  echo ">>> [1/4] SKIP vanilla — already trained (${LOG_BASE}/vanilla)"
else
  echo ">>> [1/4] Training vanilla (nominal) policies..."
  rm -rf "${LOG_BASE}/vanilla" 2>/dev/null
  for seed in $(seq 0 $((SEEDS - 1))); do
    echo "  [vanilla] seed=$seed"
    python -m src.train --env "$ENV" --mode nominal --seed "$seed" \
      --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
      --start-steps "$START_STEPS" --update-after "$UPDATE_AFTER" \
      --num-envs "$NUM_ENVS" --device "$DEVICE" \
      --log-dir "${LOG_BASE}/vanilla"
  done
fi

# ── Phase 2: RARL (adversarial, no transformer) ───────────────────────
echo ""
if method_already_trained "${LOG_BASE}/rarl" "$SEEDS"; then
  echo ">>> [2/4] SKIP rarl — already trained (${LOG_BASE}/rarl)"
else
  echo ">>> [2/4] Training RARL policies..."
  rm -rf "${LOG_BASE}/rarl" 2>/dev/null
  for seed in $(seq 0 $((SEEDS - 1))); do
    echo "  [rarl] seed=$seed"
    python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
      --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
      --start-steps "$START_STEPS" --update-after "$UPDATE_AFTER" \
      --num-envs "$NUM_ENVS" --device "$DEVICE" \
      --log-dir "${LOG_BASE}/rarl" --no-transformer \
      --disturbance-ratio 0.05 --disturbance-prob 0.3 \
      --pi-opt-path "${LOG_BASE}/vanilla/seed_${seed}/checkpoints"
  done
fi

# ── Phase 3: SA-MDP ───────────────────────────────────────────────────
echo ""
if method_already_trained "${LOG_BASE}/sa_mdp" "$SEEDS"; then
  echo ">>> [3/4] SKIP sa_mdp — already trained (${LOG_BASE}/sa_mdp)"
else
  echo ">>> [3/4] Training SA-MDP policies..."
  rm -rf "${LOG_BASE}/sa_mdp" 2>/dev/null
  for seed in $(seq 0 $((SEEDS - 1))); do
    echo "  [sa_mdp] seed=$seed"
    python -m src.baselines.sa_mdp --env "$ENV" --seed "$seed" \
      --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
      --start-steps "$START_STEPS" --update-after "$UPDATE_AFTER" \
      --num-envs "$NUM_ENVS" --device "$DEVICE" \
      --log-dir "${LOG_BASE}/sa_mdp/seed_${seed}"
  done
fi

# ── Phase 4: Domain Randomization ─────────────────────────────────────
echo ""
if method_already_trained "${LOG_BASE}/dr" "$SEEDS"; then
  echo ">>> [4/4] SKIP dr — already trained (${LOG_BASE}/dr)"
else
  echo ">>> [4/4] Training Domain Randomization policies..."
  rm -rf "${LOG_BASE}/dr" 2>/dev/null
  for seed in $(seq 0 $((SEEDS - 1))); do
    echo "  [dr] seed=$seed"
    python -m src.baselines.domain_randomization --env "$ENV" --seed "$seed" \
      --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
      --start-steps "$START_STEPS" --update-after "$UPDATE_AFTER" \
      --num-envs "$NUM_ENVS" --device "$DEVICE" \
      --log-dir "${LOG_BASE}/dr/seed_${seed}"
  done
fi

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
echo ""
echo ">>> Baseline training complete in ${TRAIN_ELAPSED}s"

# ── Phase 5: Evaluation ──────────────────────────────────────────────
echo ""
# Include RZSM in eval if it exists
METHODS="vanilla,rarl,sa_mdp,dr"
if [[ -d "${LOG_BASE}/rzsm" ]]; then
  echo ">>> Found existing RZSM checkpoints — including in evaluation."
  METHODS="vanilla,rarl,sa_mdp,dr,rzsm"
fi

echo ">>> Running evaluation ($EVAL_EPS episodes per seed)..."
python -m src.eval \
  --env "$ENV" \
  --methods "$METHODS" \
  --episodes "$EVAL_EPS" \
  --checkpoint-dir "$LOG_BASE" \
  --device "$DEVICE" \
  --output results/

# ── Phase 6: Plots ───────────────────────────────────────────────────
echo ""
echo ">>> Generating plots..."
python scripts/plot_results.py \
  --env "$ENV" \
  --log-dir logs

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TRAIN_START ))

echo ""
echo "=========================================="
echo "  Baselines done!"
echo "  Training:    ${TRAIN_ELAPSED}s"
echo "  Total:       ${TOTAL_ELAPSED}s"
echo "  Checkpoints: ${LOG_BASE}/{vanilla,rarl,sa_mdp,dr}/seed_*/"
echo "  Results:     results/${ENV_SAFE}/"
echo "=========================================="
