#!/usr/bin/env bash
# Full pipeline: train all methods, evaluate, and plot for one environment.
#
# Python-style named arguments. Default: 1M steps, 100 envs, cuda.
#
# Usage:
#   bash scripts/train_eval.sh --env HalfCheetah-v5
#   bash scripts/train_eval.sh --env Ant-v5 --steps 2000000 --seeds 3
#   bash scripts/train_eval.sh --env Walker2d-v5 --steps 500000 --device cpu
#
# Arguments:
#   --env        Gymnasium environment ID       (required)
#   --steps      Total training steps           (default: 1000000)
#   --seeds      Number of random seeds         (default: 5)
#   --num-envs   Parallel vectorized envs       (default: 100)
#   --device     Torch device                   (default: cuda)
#   --eval-eps   Eval episodes per seed         (default: 10)
#   --run        Force run number               (default: auto-increment)

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
RUN_NUM_ARG=""

# ── Parse named arguments ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)        ENV="$2";        shift 2 ;;
    --steps)      STEPS="$2";      shift 2 ;;
    --seeds)      SEEDS="$2";      shift 2 ;;
    --num-envs)   NUM_ENVS="$2";   shift 2 ;;
    --device)     DEVICE="$2";     shift 2 ;;
    --eval-eps)   EVAL_EPS="$2";   shift 2 ;;
    --run)        RUN_NUM_ARG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/train_eval.sh --env <ENV> [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --env        Gymnasium environment ID       (required)"
      echo "  --steps      Total training steps           (default: 1000000)"
      echo "  --seeds      Number of random seeds         (default: 5)"
      echo "  --num-envs   Parallel vectorized envs       (default: 100)"
      echo "  --device     Torch device                   (default: cuda)"
      echo "  --eval-eps   Eval episodes per seed         (default: 10)"
      echo "  --run        Force run number               (default: auto)"
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
  echo "Usage: bash scripts/train_eval.sh --env HalfCheetah-v5 [OPTIONS]"
  exit 1
fi

# ── Compute epochs from total steps ──────────────────────────────────
STEPS_PER_EPOCH=4000
EPOCHS=$(( STEPS / STEPS_PER_EPOCH ))
if (( EPOCHS < 1 )); then
  EPOCHS=1
fi

ENV_SAFE="${ENV//-/_}"
ENV_DIR="logs/${ENV_SAFE}"

# ── Determine run number ─────────────────────────────────────────────
mkdir -p "$ENV_DIR"
if [[ -n "$RUN_NUM_ARG" ]]; then
  RUN_NUM="$RUN_NUM_ARG"
else
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
RESULTS_DIR="results/${ENV_SAFE}"
mkdir -p "$LOG_BASE"

echo "=========================================="
echo "  Full Train + Eval Pipeline"
echo "  Env:       $ENV"
echo "  Steps:     $STEPS  (${EPOCHS} epochs x ${STEPS_PER_EPOCH})"
echo "  Seeds:     $SEEDS"
echo "  Num envs:  $NUM_ENVS"
echo "  Device:    $DEVICE"
echo "  Eval eps:  $EVAL_EPS"
echo "  Run:       $RUN_NUM"
echo "  Log dir:   $LOG_BASE"
echo "=========================================="

TRAIN_START=$(date +%s)

# ── Phase 1: Vanilla (nominal) ────────────────────────────────────────
echo ""
echo ">>> [1/5] Training vanilla (nominal) policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [vanilla] seed=$seed"
  python -m src.train --env "$ENV" --mode nominal --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" --device "$DEVICE" \
    --log-dir "${LOG_BASE}/vanilla"
done

# ── Phase 2: RARL (adversarial, no transformer) ───────────────────────
echo ""
echo ">>> [2/5] Training RARL policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [rarl] seed=$seed"
  python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" --device "$DEVICE" \
    --log-dir "${LOG_BASE}/rarl" --no-transformer \
    --disturbance-ratio 0.05 --disturbance-prob 0.3 \
    --pi-opt-path "${LOG_BASE}/vanilla/seed_${seed}/checkpoints"
done

# ── Phase 3: SA-MDP ───────────────────────────────────────────────────
echo ""
echo ">>> [3/5] Training SA-MDP policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [sa_mdp] seed=$seed"
  python -m src.baselines.sa_mdp --env "$ENV" --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" --device "$DEVICE" \
    --log-dir "${LOG_BASE}/sa_mdp/seed_${seed}"
done

# ── Phase 4: Domain Randomization ─────────────────────────────────────
echo ""
echo ">>> [4/5] Training Domain Randomization policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [dr] seed=$seed"
  python -m src.baselines.domain_randomization --env "$ENV" --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" --device "$DEVICE" \
    --log-dir "${LOG_BASE}/dr/seed_${seed}"
done

# ── Phase 5: RZSM (adversarial + transformer) ─────────────────────────
echo ""
echo ">>> [5/5] Training RZSM policies..."
for seed in $(seq 0 $((SEEDS - 1))); do
  echo "  [rzsm] seed=$seed"
  python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" --device "$DEVICE" \
    --log-dir "${LOG_BASE}/rzsm" \
    --disturbance-ratio 0.05 --disturbance-prob 0.3 \
    --pi-opt-path "${LOG_BASE}/vanilla/seed_${seed}/checkpoints"
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
echo ""
echo ">>> Training complete in ${TRAIN_ELAPSED}s"

# ── Phase 6: Evaluation ──────────────────────────────────────────────
echo ""
echo ">>> [6/7] Running evaluation ($EVAL_EPS episodes per seed)..."
python -m src.eval \
  --env "$ENV" \
  --episodes "$EVAL_EPS" \
  --checkpoint-dir "$LOG_BASE" \
  --device "$DEVICE" \
  --output results/

# ── Phase 7: Plots ───────────────────────────────────────────────────
echo ""
echo ">>> [7/7] Generating plots..."
python scripts/plot_results.py \
  --env "$ENV" \
  --log-dir logs

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TRAIN_START ))

echo ""
echo "=========================================="
echo "  All done!  (run $RUN_NUM)"
echo "  Training:  ${TRAIN_ELAPSED}s"
echo "  Total:     ${TOTAL_ELAPSED}s"
echo "  Checkpoints: ${LOG_BASE}/{vanilla,rarl,sa_mdp,dr,rzsm}/seed_*/"
echo "  CSVs:        $RESULTS_DIR/comparison_table.csv"
echo "               $RESULTS_DIR/per_seed_results.csv"
echo "  Figures:     $RESULTS_DIR/figures/"
echo "=========================================="
