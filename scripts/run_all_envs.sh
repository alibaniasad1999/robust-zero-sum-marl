#!/usr/bin/env bash
# Train all methods across ALL MuJoCo environments sequentially.
# After each env finishes, results are committed & pushed to git.
#
# Usage:
#   bash scripts/run_all_envs.sh                          # defaults
#   bash scripts/run_all_envs.sh --seeds 3 --epochs 50
#   bash scripts/run_all_envs.sh --seeds 5 --epochs 100 --num-envs 100

set -euo pipefail

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_DIR/.env/bin/activate" ]]; then
  source "$PROJECT_DIR/.env/bin/activate"
fi

# ── Defaults ────────────────────────────────────────────────────────
NUM_SEEDS=5
EPOCHS=100
NUM_ENVS=100

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
  --seeds)    NUM_SEEDS="$2"; shift 2 ;;
  --epochs)   EPOCHS="$2";    shift 2 ;;
  --num-envs) NUM_ENVS="$2";  shift 2 ;;
  *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ENVS=(
  "Ant-v5"
  "HalfCheetah-v5"
  "Hopper-v5"
  "HumanoidStandup-v5"
  "Humanoid-v5"
  "InvertedDoublePendulum-v5"
  "InvertedPendulum-v5"
  "Pusher-v5"
  "Reacher-v5"
  "Swimmer-v5"
  "Walker2d-v5"
)

TOTAL=${#ENVS[@]}
COMPLETED=0
FAILED_ENVS=()

echo "=========================================="
echo "  Full MuJoCo Benchmark"
echo "  Envs: ${TOTAL} | Seeds: ${NUM_SEEDS} | Epochs: ${EPOCHS}"
echo "  Parallel envs: ${NUM_ENVS}"
echo "=========================================="

for ENV in "${ENVS[@]}"; do
  COMPLETED=$((COMPLETED + 1))
  ENV_SAFE="${ENV//-/_}"

  echo ""
  echo "##########################################################"
  echo "  [${COMPLETED}/${TOTAL}] ${ENV}"
  echo "##########################################################"

  if ! bash "$SCRIPT_DIR/run_seeds.sh" \
    --env "$ENV" \
    --seeds "$NUM_SEEDS" \
    --epochs "$EPOCHS" \
    --num-envs "$NUM_ENVS"; then
    echo "  [WARN] ${ENV} failed — continuing to next env."
    FAILED_ENVS+=("$ENV")
  fi

  # ── Git push results for this env ──────────────────────────────
  echo ""
  echo ">>> Pushing ${ENV} results to git..."
  TIMESTAMP=$(date +"%Y-%m-%d_%H:%M")
  git add -A "logs/${ENV_SAFE}/" 2>/dev/null || true
  git add -A "results/${ENV_SAFE}/" 2>/dev/null || true
  git add -A results/ logs/ 2>/dev/null || true
  git commit -m "[${COMPLETED}/${TOTAL}] ${ENV}: ${NUM_SEEDS} seeds ${EPOCHS} epochs [${TIMESTAMP}]" \
    || echo "  Nothing new to commit for ${ENV}."
  git push || echo "  Git push failed for ${ENV} — will retry at end."

done

# ── Final push (catch anything missed) ──────────────────────────────
echo ""
echo ">>> Final git push..."
git add -A logs/ results/ 2>/dev/null || true
git commit -m "Final: all envs complete [$(date +"%Y-%m-%d_%H:%M")]" 2>/dev/null || true
git push 2>/dev/null || true

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Full Benchmark Complete!"
echo "  Passed: $((TOTAL - ${#FAILED_ENVS[@]}))/${TOTAL}"
if [[ ${#FAILED_ENVS[@]} -gt 0 ]]; then
  echo "  Failed: ${FAILED_ENVS[*]}"
fi
echo "=========================================="
