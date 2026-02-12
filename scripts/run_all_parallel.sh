#!/usr/bin/env bash
# Run ALL MuJoCo environments in PARALLEL (each in its own background process).
# After each env finishes, eval + plot + git push automatically.
#
# Usage:
#   bash scripts/run_all_parallel.sh                             # defaults: 250 epochs, 5 seeds
#   bash scripts/run_all_parallel.sh --seeds 3 --epochs 250
#   bash scripts/run_all_parallel.sh --epochs 750 --num-envs 50  # 3M steps, fewer parallel envs per job
#   bash scripts/run_all_parallel.sh --plots-only                # skip eval, just plot + push

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_DIR/.env/bin/activate" ]]; then
  source "$PROJECT_DIR/.env/bin/activate"
fi
cd "$PROJECT_DIR"

# ── Defaults ────────────────────────────────────────────────────────
NUM_SEEDS=5
EPOCHS=250
NUM_ENVS=100
PLOTS_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
  --seeds)    NUM_SEEDS="$2"; shift 2 ;;
  --epochs)   EPOCHS="$2";    shift 2 ;;
  --num-envs) NUM_ENVS="$2";  shift 2 ;;
  --plots-only) PLOTS_ONLY=true; shift ;;
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
LOG_FILE="$PROJECT_DIR/parallel_run.log"
LOCK_FILE="/tmp/rzsm_git.lock"

echo "=========================================="
echo "  Parallel Training — ${TOTAL} environments"
echo "  Seeds: $NUM_SEEDS | Epochs: $EPOCHS | Parallel envs: $NUM_ENVS"
echo "  Plots only: $PLOTS_ONLY"
echo "  Master log: $LOG_FILE"
echo "=========================================="
echo "" > "$LOG_FILE"

# ── Git helper (mutex to avoid concurrent git conflicts) ────────────
git_push_results() {
  local env_id="$1"
  local env_safe="${env_id//-/_}"

  # Simple file-based lock
  while ! mkdir "$LOCK_FILE" 2>/dev/null; do
    sleep $((RANDOM % 5 + 1))
  done
  trap "rmdir '$LOCK_FILE' 2>/dev/null" RETURN

  local ts
  ts=$(date +"%Y-%m-%d_%H:%M")
  git add -A "logs/${env_safe}/" 2>/dev/null || true
  git add -A "results/${env_safe}/" 2>/dev/null || true
  git add -A results/ 2>/dev/null || true
  git commit -m "${env_id}: ${NUM_SEEDS} seeds ${EPOCHS} epochs [${ts}]" 2>/dev/null || true
  git push 2>/dev/null || true

  rmdir "$LOCK_FILE" 2>/dev/null || true
  trap - RETURN
}

# ── Per-env job function ────────────────────────────────────────────
run_env() {
  local env_id="$1"
  local env_safe="${env_id//-/_}"
  local job_log="$PROJECT_DIR/logs/${env_safe}_parallel.log"
  mkdir -p "$PROJECT_DIR/logs"

  echo "[$(date +%H:%M:%S)] START  $env_id" >> "$LOG_FILE"

  if [[ "$PLOTS_ONLY" == "false" ]]; then
    # Train all methods
    bash "$SCRIPT_DIR/run_seeds.sh" \
      --env "$env_id" \
      --seeds "$NUM_SEEDS" \
      --epochs "$EPOCHS" \
      --num-envs "$NUM_ENVS" \
      > "$job_log" 2>&1
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
      echo "[$(date +%H:%M:%S)] FAILED $env_id (exit=$exit_code)" >> "$LOG_FILE"
      echo "FAILED" > "/tmp/rzsm_status_${env_safe}"
      return $exit_code
    fi
  else
    # Plots only — just run eval + plot
    python -m src.eval --env "$env_id" \
      --checkpoint-dir "logs/${env_safe}/" --episodes 50 \
      > "$job_log" 2>&1 || true
    python scripts/plot_results.py --env "$env_id" --log-dir logs \
      >> "$job_log" 2>&1 || true
  fi

  # Push this env's results
  git_push_results "$env_id"

  echo "[$(date +%H:%M:%S)] DONE   $env_id" >> "$LOG_FILE"
  echo "DONE" > "/tmp/rzsm_status_${env_safe}"
}

# ── Launch all envs in parallel ─────────────────────────────────────
rmdir "$LOCK_FILE" 2>/dev/null || true
PIDS=()

for ENV in "${ENVS[@]}"; do
  env_safe="${ENV//-/_}"
  echo "RUNNING" > "/tmp/rzsm_status_${env_safe}"
  run_env "$ENV" &
  PIDS+=($!)
  echo "  Launched: $ENV (PID ${PIDS[-1]})"
done

echo ""
echo "  All ${TOTAL} jobs launched. Monitoring progress..."
echo ""

# ── Monitor loop ────────────────────────────────────────────────────
while true; do
  done_count=0
  running=""
  for i in "${!ENVS[@]}"; do
    env_safe="${ENVS[$i]//-/_}"
    status=$(cat "/tmp/rzsm_status_${env_safe}" 2>/dev/null || echo "UNKNOWN")
    if [[ "$status" == "DONE" || "$status" == "FAILED" ]]; then
      done_count=$((done_count + 1))
    else
      running+=" ${ENVS[$i]}"
    fi
  done

  pct=$((done_count * 100 / TOTAL))
  printf "\r  Progress: %d/%d (%d%%)  Running:%s     " "$done_count" "$TOTAL" "$pct" "$running"

  if [[ $done_count -ge $TOTAL ]]; then
    break
  fi
  sleep 10
done

echo ""

# ── Wait for all background processes ───────────────────────────────
for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

# ── Final git push ──────────────────────────────────────────────────
echo ""
echo ">>> Final git push..."
git add -A results/ logs/ 2>/dev/null || true
git commit -m "All envs complete: ${TOTAL} envs ${NUM_SEEDS} seeds ${EPOCHS} epochs [$(date +%Y-%m-%d_%H:%M)]" 2>/dev/null || true
git push 2>/dev/null || true

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
for ENV in "${ENVS[@]}"; do
  env_safe="${ENV//-/_}"
  status=$(cat "/tmp/rzsm_status_${env_safe}" 2>/dev/null || echo "UNKNOWN")
  printf "  %-35s %s\n" "$ENV" "$status"
done
echo ""
echo "  Master log: $LOG_FILE"
echo "  Per-env logs: logs/*_parallel.log"
echo "=========================================="

# Cleanup
for ENV in "${ENVS[@]}"; do
  rm -f "/tmp/rzsm_status_${ENV//-/_}"
done
rmdir "$LOCK_FILE" 2>/dev/null || true
