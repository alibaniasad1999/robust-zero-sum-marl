#!/usr/bin/env bash
# Evaluate all methods and generate plots + CSVs for one environment.
#
# Usage:
#   bash scripts/eval_and_plot.sh Ant-v5
#   bash scripts/eval_and_plot.sh HalfCheetah-v5 100
#   bash scripts/eval_and_plot.sh Walker2d-v5 50
#
# Arguments:
#   $1  ENV        Gymnasium environment ID  (default: Ant-v5)
#   $2  EPISODES   Evaluation episodes per (method, scenario, seed)  (default: 50)

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
EPISODES="${2:-50}"

ENV_SAFE="${ENV//-/_}"
LOG_DIR="logs/${ENV_SAFE}"
RESULTS_DIR="results/${ENV_SAFE}"

echo "=========================================="
echo "  Evaluation & Plotting Pipeline"
echo "  Env:       $ENV"
echo "  Episodes:  $EPISODES per (method, scenario, seed)"
echo "  Log dir:   $LOG_DIR"
echo "  Output:    $RESULTS_DIR"
echo "=========================================="

# ── Check that logs exist ─────────────────────────────────────────────
if [[ ! -d "$LOG_DIR" ]]; then
  echo "ERROR: Log directory '$LOG_DIR' not found."
  echo "  Train first:  bash scripts/run.sh $ENV"
  exit 1
fi

# ── Phase 1: Evaluation ──────────────────────────────────────────────
echo ""
echo ">>> [1/2] Running evaluation..."
python -m src.eval \
  --env "$ENV" \
  --episodes "$EPISODES" \
  --checkpoint-dir "$LOG_DIR" \
  --output results/

# ── Phase 2: Plots ───────────────────────────────────────────────────
echo ""
echo ">>> [2/2] Generating plots..."
python scripts/plot_results.py \
  --env "$ENV" \
  --log-dir logs

echo ""
echo "=========================================="
echo "  Done!"
echo "  CSVs:    $RESULTS_DIR/comparison_table.csv"
echo "           $RESULTS_DIR/per_seed_results.csv"
echo "  Figures: $RESULTS_DIR/figures/"
echo "=========================================="
