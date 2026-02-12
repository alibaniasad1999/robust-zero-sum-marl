#!/usr/bin/env bash
# Quick sanity check: run 1 epoch, 1 seed, 1 env on every MuJoCo environment
# to detect which ones produce NaN returns.
#
# Usage:
#   bash scripts/check_nan_envs.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_DIR/.env/bin/activate" ]]; then
  source "$PROJECT_DIR/.env/bin/activate"
fi

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

TMP_DIR="/tmp/nan_check_$$"
mkdir -p "$TMP_DIR"

PASS_LIST=()
NAN_LIST=()
ERR_LIST=()

echo "=========================================="
echo "  NaN Check — 1 epoch, 1 seed, 1 env each"
echo "  Temp logs: $TMP_DIR"
echo "=========================================="

for ENV in "${ENVS[@]}"; do
  ENV_SAFE="${ENV//-/_}"
  LOG_DIR="$TMP_DIR/$ENV_SAFE"
  printf "\n  %-35s " "$ENV"

  # Run vanilla training only: 1 epoch, 1 seed, 1 parallel env
  OUTPUT=$(python -m src.train \
    --env "$ENV" --mode nominal --seed 0 \
    --epochs 1 --num-envs 1 \
    --log-dir "$LOG_DIR/vanilla" 2>&1)

  EXIT_CODE=$?

  # Check for actual NaN return values in output (not code logic strings)
  HAS_NAN=false
  if echo "$OUTPUT" | grep -qiE 'return[= ]+nan|ret=nan|mean_return.*nan'; then
    HAS_NAN=true
  fi

  # Check the training_returns.csv for nan values in data rows
  CSV="$LOG_DIR/vanilla/seed_0/training_returns.csv"
  CSV_NAN=false
  if [[ -f "$CSV" ]]; then
    if tail -n +2 "$CSV" | grep -qi "nan"; then
      CSV_NAN=true
    fi
  fi

  # Check network params for NaN via quick python check
  PARAM_NAN=false
  CKPT="$LOG_DIR/vanilla/seed_0/checkpoints/pi.pt"
  if [[ -f "$CKPT" ]]; then
    PARAM_NAN=$(python -c "
import torch
sd = torch.load('$CKPT', map_location='cpu', weights_only=True)
has_nan = any(torch.isnan(v).any().item() for v in sd.values())
print(has_nan)
" 2>/dev/null || echo "False")
  fi

  # Save full log
  echo "$OUTPUT" > "$TMP_DIR/${ENV_SAFE}_log.txt"

  # Determine status
  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR (exit=$EXIT_CODE)"
    ERR_LIST+=("$ENV")
  elif [[ "$HAS_NAN" == "true" || "$CSV_NAN" == "true" || "$PARAM_NAN" == "True" ]]; then
    echo "NaN DETECTED  (output=$HAS_NAN csv=$CSV_NAN params=$PARAM_NAN)"
    NAN_LIST+=("$ENV")
  else
    # Show last return from CSV
    LAST_RET=""
    if [[ -f "$CSV" ]]; then
      LAST_RET=$(tail -1 "$CSV" | cut -d',' -f2)
    fi
    echo "OK  (last_return=${LAST_RET:-n/a})"
    PASS_LIST+=("$ENV")
  fi
done

echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
echo "  Passed (${#PASS_LIST[@]}): ${PASS_LIST[*]:-none}"
echo "  NaN    (${#NAN_LIST[@]}): ${NAN_LIST[*]:-none}"
echo "  Error  (${#ERR_LIST[@]}): ${ERR_LIST[*]:-none}"
echo ""
echo "  Full logs: $TMP_DIR/"
echo "=========================================="

if [[ ${#NAN_LIST[@]} -gt 0 || ${#ERR_LIST[@]} -gt 0 ]]; then
  echo ""
  echo "  Showing last 15 lines of failed env logs:"
  for ENV in "${NAN_LIST[@]}" "${ERR_LIST[@]}"; do
    ENV_SAFE="${ENV//-/_}"
    echo ""
    echo "  --- $ENV ---"
    tail -15 "$TMP_DIR/${ENV_SAFE}_log.txt" | sed 's/^/    /'
  done
  exit 1
fi
