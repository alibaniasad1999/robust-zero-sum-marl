#!/usr/bin/env bash
# Train all methods across multiple seeds.
#
# Usage:
#   bash scripts/run_seeds.sh                    # default: 5 seeds, 250 epochs (1M steps)
#   bash scripts/run_seeds.sh --seeds 3 --epochs 125  # 500K steps
#   bash scripts/run_seeds.sh --env Humanoid-v5 --epochs 750  # 3M steps (paper-standard for hard envs)
#   bash scripts/run_seeds.sh --seeds 5 --epochs 250 --num-envs 200  # faster with more parallel envs

set -euo pipefail

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_DIR/.env/bin/activate" ]]; then
  source "$PROJECT_DIR/.env/bin/activate"
fi

# ── Defaults ────────────────────────────────────────────────────────
# Paper-standard: 1M total transitions (steps_per_epoch=4000 × epochs=250).
# For harder envs (Ant, Humanoid, HumanoidStandup) use --epochs 750 (3M steps).
# num_envs only affects speed (parallelism), NOT total transitions.
NUM_SEEDS=5
EPOCHS=250
NUM_ENVS=10
ENV="Ant-v5"

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
  --seeds)
    NUM_SEEDS="$2"
    shift 2
    ;;
  --epochs)
    EPOCHS="$2"
    shift 2
    ;;
  --num-envs)
    NUM_ENVS="$2"
    shift 2
    ;;
  --env)
    ENV="$2"
    shift 2
    ;;
  *)
    echo "Unknown arg: $1"
    exit 1
    ;;
  esac
done

# ── Sanitize env name for directory paths ───────────────────────────
ENV_SAFE="${ENV//-/_}"   # e.g. Ant-v5 → Ant_v5
LOG_BASE="logs/${ENV_SAFE}"

echo "=========================================="
echo "  Multi-seed training"
echo "  Seeds: $NUM_SEEDS | Epochs: $EPOCHS"
echo "  Env: $ENV | Parallel envs: $NUM_ENVS"
echo "  Log base: $LOG_BASE"
echo "=========================================="

# ── Phase 1: Vanilla (nominal) ──────────────────────────────────────
echo ""
echo ">>> Phase 1: Training vanilla (nominal) policies..."
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
  echo "  [vanilla] seed=$seed"
  python -m src.train --env "$ENV" --mode nominal --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/vanilla"
done

# ── Phase 2: RARL (adversarial, no transformer) ─────────────────────
echo ""
echo ">>> Phase 2: Training RARL policies..."
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
  echo "  [rarl] seed=$seed"
  python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/rarl" --no-transformer \
    --pi-opt-path "${LOG_BASE}/vanilla/seed_${seed}/checkpoints"
done

# ── Phase 3: SA-MDP ─────────────────────────────────────────────────
echo ""
echo ">>> Phase 3: Training SA-MDP policies..."
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
  echo "  [sa_mdp] seed=$seed"
  python -m src.baselines.sa_mdp --env "$ENV" --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/sa_mdp/seed_${seed}"
done

# ── Phase 4: Domain Randomization ───────────────────────────────────
echo ""
echo ">>> Phase 4: Training Domain Randomization policies..."
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
  echo "  [dr] seed=$seed"
  python -m src.baselines.domain_randomization --env "$ENV" --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/dr/seed_${seed}"
done

# ── Phase 5: RZSM (adversarial + transformer) ───────────────────────
echo ""
echo ">>> Phase 5: Training RZSM policies..."
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
  echo "  [rzsm] seed=$seed"
  python -m src.train --env "$ENV" --mode adversarial --seed "$seed" \
    --epochs "$EPOCHS" --num-envs "$NUM_ENVS" \
    --log-dir "${LOG_BASE}/rzsm" \
    --pi-opt-path "${LOG_BASE}/vanilla/seed_${seed}/checkpoints"
done

echo ""
echo "=========================================="
echo "  Training complete!"
echo "  Checkpoints: ${LOG_BASE}/{vanilla,rarl,sa_mdp,dr,rzsm}/seed_*/"
echo "=========================================="

# ── Phase 6: Evaluate ───────────────────────────────────────────────
echo ""
echo ">>> Phase 6: Running evaluation..."
python -m src.eval --env "$ENV" --checkpoint-dir "${LOG_BASE}/" --episodes 50

# ── Phase 7: Plot results ───────────────────────────────────────────
echo ""
echo ">>> Phase 7: Generating plots..."
python scripts/plot_results.py --env "$ENV" --log-dir logs

# ── Phase 8: Git push results ───────────────────────────────────────
echo ""
echo ">>> Phase 8: Pushing results to git..."
TIMESTAMP=$(date +"%Y-%m-%d_%H:%M")
git add -A "logs/${ENV_SAFE}/" "results/${ENV_SAFE}/" "results/figures/" 2>/dev/null || true
git add -A results/ logs/ 2>/dev/null || true
git commit -m "Training results: ${ENV} ${NUM_SEEDS} seeds ${EPOCHS} epochs [${TIMESTAMP}]" || echo "  Nothing to commit."
git push || echo "  Git push failed — check remote config."

echo ""
echo "=========================================="
echo "  All done! Results committed & pushed."
echo "=========================================="
