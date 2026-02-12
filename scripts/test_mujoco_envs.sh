#!/usr/bin/env bash
# Quick smoke test for all supported MuJoCo environments.
#
# Usage:
#   bash scripts/test_mujoco_envs.sh
#   bash scripts/test_mujoco_envs.sh --steps 50

set -euo pipefail

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_DIR/.env/bin/activate" ]]; then
  source "$PROJECT_DIR/.env/bin/activate"
fi

STEPS=20

while [[ $# -gt 0 ]]; do
  case $1 in
  --steps) STEPS="$2"; shift 2 ;;
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

PASSED=0
FAILED=0
FAIL_LIST=()

echo "=========================================="
echo "  MuJoCo Environment Smoke Test"
echo "  Steps per env: $STEPS"
echo "=========================================="
echo ""

for env in "${ENVS[@]}"; do
  printf "  %-35s" "$env"
  if python -c "
import gymnasium as gym
import numpy as np

env = gym.make('$env')
obs, info = env.reset()
assert obs is not None, 'reset returned None obs'
total_reward = 0.0
for _ in range($STEPS):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print(f'obs_dim={obs.shape[0]}  act_dim={env.action_space.shape[0]}  reward={total_reward:.2f}')
" 2>/dev/null; then
    PASSED=$((PASSED + 1))
  else
    echo "FAILED"
    FAILED=$((FAILED + 1))
    FAIL_LIST+=("$env")
  fi
done

echo ""
echo "=========================================="
echo "  Results: $PASSED passed, $FAILED failed"
if [[ $FAILED -gt 0 ]]; then
  echo "  Failed: ${FAIL_LIST[*]}"
  echo "=========================================="
  exit 1
fi
echo "=========================================="
