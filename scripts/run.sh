#!/usr/bin/env bash
# Simple training runner — train all methods for one environment.
#
# Logs are saved with auto-incrementing run numbers:
#   logs/Ant_v5/run_0/{vanilla,rarl,...}/seed_*/
#   logs/Ant_v5/run_1/{vanilla,rarl,...}/seed_*/
#
# Usage:
#   bash scripts/run.sh Ant-v5 250 5 10        # auto-increments run number
#   bash scripts/run.sh HalfCheetah-v5 100 3 20 2  # forces run_2
#
# Arguments:
#   $1  ENV       Gymnasium environment ID  (default: Ant-v5)
#   $2  EPOCHS    Number of training epochs  (default: 250)
#   $3  SEEDS     Number of random seeds     (default: 5)
#   $4  NUM_ENVS  Parallel vectorized envs   (default: 10)
#   $5  RUN_NUM   Run number (default: auto-increment)

set -euo pipefail

# ── Activate virtual environment ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_DIR/.env/bin/activate" ]]; then
  source "$PROJECT_DIR/.env/bin/activate"
fi
cd "$PROJECT_DIR"

# ── Arguments ─────────────────────────────────────────────────────────
ENV="${1:-Ant-v5}"
EPOCHS="${2:-250}"
SEEDS="${3:-5}"
NUM_ENVS="${4:-10}"
RUN_NUM_ARG="${5:-}"

ENV_SAFE="${ENV//-/_}"
ENV_DIR="logs/${ENV_SAFE}"

# ── Determine run number ────────────────────────────────────────────────
mkdir -p "$ENV_DIR"
if [[ -n "$RUN_NUM_ARG" ]]; then
  # User specified a run number
  RUN_NUM="$RUN_NUM_ARG"
else
  # Auto-increment: find next available
  RUN_NUM=0
  for d in "$ENV_DIR"/run_*; do
    if [[ -d "$d" ]]; then
      n="${d##*run_}"
      if [[ "$n" =~ ^[0-9]+$ ]] && (( n >= RUN_NUM )); then
        RUN_NUM=$(( n + 1 ))
      fi
    fi
  done
fi

LOG_BASE="${ENV_DIR}/run_${RUN_NUM}"
mkdir -p "$LOG_BASE"

echo "=========================================="
echo "  Training Pipeline"
echo "  Env:      $ENV"
echo "  Epochs:   $EPOCHS  ($(( EPOCHS * 4000 )) total steps)"
echo "  Seeds:    $SEEDS"
echo "  Num envs: $NUM_ENVS"
echo "  Run:      $RUN_NUM"
echo "  Log dir:  $LOG_BASE"
echo "=========================================="

# ── Phase 1: Vanilla (nominal) ────────────────────────────────────────
echo ""
echo ">>> [1/5] Training vanilla (nominal) policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [vanilla] seed=$seed"
  python -m src.train --env "$ENV" --mode nominal --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/vanilla"
done

# ── Phase 2: RARL (adversarial, no transformer) ───────────────────────
echo ""
echo ">>> [2/5] Training RARL policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [rarl] seed=$seed"
  python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/rarl" --no-transformer \
    --pi-opt-path "${LOG_BASE}/vanilla/seed_${seed}/checkpoints"
done

# ── Phase 3: SA-MDP ───────────────────────────────────────────────────
echo ""
echo ">>> [3/5] Training SA-MDP policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [sa_mdp] seed=$seed"
  python -m src.baselines.sa_mdp --env "$ENV" --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/sa_mdp/seed_${seed}"
done

# ── Phase 4: Domain Randomization ─────────────────────────────────────
echo ""
echo ">>> [4/5] Training Domain Randomization policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [dr] seed=$seed"
  python -m src.baselines.domain_randomization --env "$ENV" --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/dr/seed_${seed}"
done

# ── Phase 5: RZSM (adversarial + transformer) ─────────────────────────
echo ""
echo ">>> [5/5] Training RZSM policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [rzsm] seed=$seed"
  python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/rzsm" \
    --pi-opt-path "${LOG_BASE}/vanilla/seed_${seed}/checkpoints"
done

echo ""
echo "=========================================="
echo "  Training complete!  (run $RUN_NUM)"
echo "  Checkpoints: ${LOG_BASE}/{vanilla,rarl,sa_mdp,dr,rzsm}/seed_*/"
echo "=========================================="
