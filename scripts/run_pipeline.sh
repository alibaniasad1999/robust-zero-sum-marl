#!/usr/bin/env bash
# =============================================================================
#  run_pipeline.sh — Full pipeline: train → tune → RZSM best → evaluate
#
#  One command does everything (default: Ant-v5):
#    bash scripts/run_pipeline.sh
#
#  Select environment:
#    bash scripts/run_pipeline.sh --env HalfCheetah-v5
#    bash scripts/run_pipeline.sh --env Humanoid-v5
#
#  Other flags:
#    --tune-trials 20
#    --skip-baselines
#    --skip-nominal
#    --skip-adversarial
#    --skip-transformer
#    --skip-tune
#    --skip-rzsm
#    --skip-eval
#    --num-envs 50
#    --device cpu
#    --seed 1
# =============================================================================
set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
ENV="Ant-v5"
SEED="0"
DEVICE="auto"
NUM_ENVS="100"
TUNE_TRIALS="40"
SKIP_BASELINES="0"
SKIP_NOMINAL="0"
SKIP_ADVERSARIAL="0"
SKIP_TRANSFORMER="0"
SKIP_TUNE="0"
SKIP_RZSM="0"
SKIP_EVAL="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)              ENV="$2";          shift 2 ;;
        --seed)             SEED="$2";         shift 2 ;;
        --device)           DEVICE="$2";       shift 2 ;;
        --num-envs)         NUM_ENVS="$2";     shift 2 ;;
        --tune-trials)      TUNE_TRIALS="$2";  shift 2 ;;
        --skip-baselines)   SKIP_BASELINES="1"; shift ;;
        --skip-nominal)     SKIP_NOMINAL="1";   shift ;;
        --skip-adversarial) SKIP_ADVERSARIAL="1"; shift ;;
        --skip-transformer) SKIP_TRANSFORMER="1"; shift ;;
        --skip-tune)        SKIP_TUNE="1";      shift ;;
        --skip-rzsm)        SKIP_RZSM="1";      shift ;;
        --skip-eval)        SKIP_EVAL="1";      shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Fixed configuration ────────────────────────────────────────────────────────
# 1 million steps = 250 epochs × 4 000 steps/epoch
STEPS_PER_EPOCH=4000
NOMINAL_EPOCHS=250          # 1 000 000 nominal steps
BASELINE_EPOCHS=250         # same for SA-MDP / DR
ADV_EPOCHS=250              # 1 000 000 adversarial steps
TRANSFORMER_EPOCHS=50       # offline detector training
TUNE_EPOCHS=30              # short epochs per Optuna trial
FINETUNE_EPOCHS=250         # RZSM final training with best params (1 M steps)
EVAL_EPISODES=50

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_DIR/.env/bin/activate"
ENV_SAFE="${ENV//-/_}"
LOGS="$REPO_DIR/logs/$ENV_SAFE"
RESULTS="$REPO_DIR/results/$ENV_SAFE"

NOMINAL_CKPT="$LOGS/nominal/seed_$SEED/checkpoints"
ADV_CKPT="$LOGS/adversarial/seed_$SEED/checkpoints"
RZSM_CKPT="$LOGS/rzsm_best/seed_$SEED/checkpoints"
DETECTOR_PT="$LOGS/detector_offline.pt"
TUNE_DB="$LOGS/tuning/optuna.db"
BEST_PARAMS_FILE="$LOGS/tuning/best_params.env"

# ── Helpers ────────────────────────────────────────────────────────────────────
BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'; RESET='\033[0m'

stage() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"; \
          echo -e "${BOLD}${CYAN}  $1${RESET}"; \
          echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"; }
done_()  { echo -e "${GREEN}  ✓ $1${RESET}"; }
skip_()  { echo -e "  ↷ Skipping: $1"; }

# Activate venv and cd to repo root (python -m src.* needs it)
source "$VENV"
cd "$REPO_DIR"
PY="$(which python)"

START_TIME=$(date +%s)

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  RZSM Full Pipeline — $ENV${RESET}"
echo -e "${BOLD}║  1 000 000 steps → tune → RZSM best → eval${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo "  env=$ENV  seed=$SEED  device=$DEVICE  num_envs=$NUM_ENVS"
echo "  nominal_epochs=$NOMINAL_EPOCHS (${STEPS_PER_EPOCH}×${NOMINAL_EPOCHS} = $(( NOMINAL_EPOCHS * STEPS_PER_EPOCH )) steps)"
echo "  tune_trials=$TUNE_TRIALS × $TUNE_EPOCHS epochs/trial"
echo ""

mkdir -p "$LOGS" "$RESULTS"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Baselines: SA-MDP + Domain Randomisation
# ─────────────────────────────────────────────────────────────────────────────
stage "1/8  Baselines (SA-MDP + Domain Randomisation)"
if [[ "$SKIP_BASELINES" == "1" ]]; then
    skip_ "baselines"
else
    $PY -m src.baselines.sa_mdp \
        --env "$ENV" \
        --epochs $BASELINE_EPOCHS \
        --steps-per-epoch $STEPS_PER_EPOCH \
        --num-envs $NUM_ENVS \
        --seed $SEED \
        --device "$DEVICE" \
        --log-dir "$LOGS/sa_mdp/seed_$SEED"
    done_ "SA-MDP"

    $PY -m src.baselines.domain_randomization \
        --env "$ENV" \
        --epochs $BASELINE_EPOCHS \
        --steps-per-epoch $STEPS_PER_EPOCH \
        --num-envs $NUM_ENVS \
        --seed $SEED \
        --device "$DEVICE" \
        --log-dir "$LOGS/domain_rand/seed_$SEED"
    done_ "Domain Randomisation"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Nominal TD3 (pi_opt)   — 1 M steps
# ─────────────────────────────────────────────────────────────────────────────
stage "2/8  Nominal TD3 — pi_opt  (1 000 000 steps)"
if [[ "$SKIP_NOMINAL" == "1" ]]; then
    skip_ "nominal TD3"
else
    $PY -m src.train \
        --env "$ENV" \
        --mode nominal \
        --epochs $NOMINAL_EPOCHS \
        --steps-per-epoch $STEPS_PER_EPOCH \
        --num-envs $NUM_ENVS \
        --seed $SEED \
        --device "$DEVICE" \
        --log-dir "$LOGS/nominal/seed_$SEED"
    done_ "Nominal TD3 → $NOMINAL_CKPT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Adversarial TD3 (pi_rob + pi_adv)   — 1 M steps
# ─────────────────────────────────────────────────────────────────────────────
stage "3/8  Adversarial TD3 — pi_rob + pi_adv  (1 000 000 steps)"
if [[ "$SKIP_ADVERSARIAL" == "1" ]]; then
    skip_ "adversarial TD3"
else
    $PY -m src.train \
        --env "$ENV" \
        --mode adversarial \
        --epochs $ADV_EPOCHS \
        --steps-per-epoch $STEPS_PER_EPOCH \
        --num-envs $NUM_ENVS \
        --seed $SEED \
        --device "$DEVICE" \
        --log-dir "$LOGS/adversarial/seed_$SEED" \
        --pi-opt-path "$NOMINAL_CKPT" \
        --disturbance-ratio 0.05 \
        --disturbance-prob 0.3 \
        --warmup-fraction 0.2
    done_ "Adversarial TD3 → $ADV_CKPT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Offline Transformer training on ALL logged data
# ─────────────────────────────────────────────────────────────────────────────
stage "4/8  Offline Transformer — aggregate all dataset dirs"
if [[ "$SKIP_TRANSFORMER" == "1" ]]; then
    skip_ "offline transformer"
else
    # Collect every dataset/ directory that was populated
    DATASET_DIRS=()
    for MODE in nominal adversarial sa_mdp domain_rand; do
        D="$LOGS/$MODE/seed_$SEED/dataset"
        [[ -d "$D/episodes" ]] && DATASET_DIRS+=("$D")
    done

    if [[ ${#DATASET_DIRS[@]} -eq 0 ]]; then
        echo "  [warn] No dataset directories found — skipping offline transformer."
    else
        echo "  Dataset dirs: ${DATASET_DIRS[*]}"
        # Infer obs_dim from the environment at runtime
        OBS_DIM=$($PY -c "import gymnasium as gym; e=gym.make('$ENV'); print(e.observation_space.shape[0]); e.close()")
        echo "  obs_dim=$OBS_DIM"

        $PY "$REPO_DIR/scripts/train_transformer.py" \
            --log-dirs "${DATASET_DIRS[@]}" \
            --obs-dim "$OBS_DIM" \
            --output "$DETECTOR_PT" \
            --epochs $TRANSFORMER_EPOCHS \
            --batch-size 512 \
            --device "$DEVICE"
        done_ "Offline Transformer → $DETECTOR_PT"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Hyperparameter Tuning  (Phase A: core  |  Phase B: transformer)
# ─────────────────────────────────────────────────────────────────────────────
stage "5/8  Optuna Hyperparameter Tuning  ($TUNE_TRIALS trials × 2 phases)"
if [[ "$SKIP_TUNE" == "1" ]]; then
    skip_ "hyperparameter tuning"
else
    $PY "$REPO_DIR/scripts/tune_rzsm.py" \
        --env "$ENV" \
        --phase both \
        --trials $TUNE_TRIALS \
        --tune-epochs $TUNE_EPOCHS \
        --steps-per-epoch $STEPS_PER_EPOCH \
        --start-steps 5000 \
        --eval-episodes 5 \
        --seed $SEED \
        --device "$DEVICE" \
        --pi-opt-path "$NOMINAL_CKPT"
    done_ "Tuning complete  →  $TUNE_DB"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — Extract best params from Optuna DB → shell variables
# ─────────────────────────────────────────────────────────────────────────────
stage "6/8  Extract best hyperparameters from Optuna DB"

$PY - <<PYEOF
import json, sys, os
try:
    import optuna
    safe = "$ENV_SAFE"
    storage = f"sqlite:///logs/{safe}/tuning/optuna.db"
    params = {}
    for phase in ("phaseA", "phaseB"):
        name = f"rzsm_{safe}_{phase}"
        try:
            s = optuna.load_study(study_name=name, storage=storage)
            params.update(s.best_params)
            print(f"  {phase}: best score={s.best_value:.2f}  trial={s.best_trial.number}")
        except Exception as e:
            print(f"  {phase}: not found ({e}), using defaults")
    # Write as a shell-sourceable file
    lines = []
    lines.append(f"BEST_HIDDEN=\"{params.get('hidden_sizes','256,256')}\"")
    lines.append(f"BEST_PI_LR=\"{params.get('pi_lr', 3e-4):.2e}\"")
    lines.append(f"BEST_Q_LR=\"{params.get('q_lr', 3e-4):.2e}\"")
    lines.append(f"BEST_GAMMA=\"{params.get('gamma', 0.99):.4f}\"")
    lines.append(f"BEST_POLYAK=\"{params.get('polyak', 0.995):.4f}\"")
    lines.append(f"BEST_BATCH=\"{params.get('batch_size', 256)}\"")
    lines.append(f"BEST_ACT_NOISE=\"{params.get('act_noise', 0.1):.3f}\"")
    lines.append(f"BEST_POLICY_DELAY=\"{params.get('policy_delay', 2)}\"")
    lines.append(f"BEST_TARGET_NOISE=\"{params.get('target_noise', 0.2):.3f}\"")
    lines.append(f"BEST_NOISE_CLIP=\"{params.get('noise_clip', 0.5):.3f}\"")
    lines.append(f"BEST_N_UPDATES=\"{params.get('n_updates', 200)}\"")
    lines.append(f"BEST_UPDATE_EVERY=\"{params.get('update_every', 50)}\"")
    lines.append(f"BEST_DIST_RATIO=\"{params.get('disturbance_ratio', 0.05):.4f}\"")
    lines.append(f"BEST_DIST_PROB=\"{params.get('disturbance_prob', 0.3):.3f}\"")
    lines.append(f"BEST_WARMUP=\"{params.get('warmup_fraction', 0.2):.3f}\"")
    lines.append(f"BEST_SEQ_LEN=\"{params.get('seq_len', 20)}\"")
    lines.append(f"BEST_D_MODEL=\"{params.get('d_model', 128)}\"")
    lines.append(f"BEST_NHEAD=\"{params.get('nhead', 4)}\"")
    lines.append(f"BEST_LAYERS=\"{params.get('num_layers', 3)}\"")
    lines.append(f"BEST_DET_LR=\"{params.get('detector_lr', 1e-4):.2e}\"")
    lines.append(f"BEST_TRAIN_INTERVAL=\"{params.get('train_interval', 200)}\"")
    os.makedirs(os.path.dirname("$BEST_PARAMS_FILE"), exist_ok=True)
    with open("$BEST_PARAMS_FILE", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved best params → $BEST_PARAMS_FILE")
except Exception as e:
    print(f"  [warn] Could not load Optuna DB: {e}")
    print(f"  Using default params.")
    with open("$BEST_PARAMS_FILE", "w") as f:
        f.write("""BEST_HIDDEN="256,256"
BEST_PI_LR="3e-04"
BEST_Q_LR="3e-04"
BEST_GAMMA="0.9900"
BEST_POLYAK="0.9950"
BEST_BATCH="256"
BEST_ACT_NOISE="0.100"
BEST_POLICY_DELAY="2"
BEST_TARGET_NOISE="0.200"
BEST_NOISE_CLIP="0.500"
BEST_N_UPDATES="200"
BEST_UPDATE_EVERY="50"
BEST_DIST_RATIO="0.0500"
BEST_DIST_PROB="0.300"
BEST_WARMUP="0.200"
BEST_SEQ_LEN="20"
BEST_D_MODEL="128"
BEST_NHEAD="4"
BEST_LAYERS="3"
BEST_DET_LR="1e-04"
BEST_TRAIN_INTERVAL="200"
""")
PYEOF

# Load best params into this shell
source "$BEST_PARAMS_FILE"
echo "  Hidden=$BEST_HIDDEN  pi_lr=$BEST_PI_LR  gamma=$BEST_GAMMA"
echo "  dist_ratio=$BEST_DIST_RATIO  seq_len=$BEST_SEQ_LEN  d_model=$BEST_D_MODEL"
done_ "Best params loaded"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — RZSM final training with best hyperparameters  (1 M steps)
# ─────────────────────────────────────────────────────────────────────────────
stage "7/8  RZSM Final Training with best params  (1 000 000 steps)"
if [[ "$SKIP_RZSM" == "1" ]]; then
    skip_ "RZSM final training"
else
    $PY -m src.train \
        --env "$ENV" \
        --mode adversarial \
        --epochs $FINETUNE_EPOCHS \
        --steps-per-epoch $STEPS_PER_EPOCH \
        --num-envs $NUM_ENVS \
        --seed $SEED \
        --device "$DEVICE" \
        --log-dir "$LOGS/rzsm_best/seed_$SEED" \
        --pi-opt-path "$NOMINAL_CKPT" \
        --hidden-sizes "$BEST_HIDDEN" \
        --batch-size "$BEST_BATCH" \
        --disturbance-ratio "$BEST_DIST_RATIO" \
        --disturbance-prob "$BEST_DIST_PROB" \
        --warmup-fraction "$BEST_WARMUP" \
        --seq-len "$BEST_SEQ_LEN" \
        --d-model "$BEST_D_MODEL" \
        --nhead "$BEST_NHEAD" \
        --transformer-layers "$BEST_LAYERS" \
        --detector-lr "$BEST_DET_LR" \
        --detector-train-interval "$BEST_TRAIN_INTERVAL"
    done_ "RZSM → $RZSM_CKPT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — Evaluation: all methods, all scenarios
# ─────────────────────────────────────────────────────────────────────────────
stage "8/8  Evaluation  ($EVAL_EPISODES episodes × 5 scenarios)"
if [[ "$SKIP_EVAL" == "1" ]]; then
    skip_ "evaluation"
else
    mkdir -p "$RESULTS"

    # Build --checkpoint-dirs so eval finds each method in the right place
    CKPT_DIRS="vanilla=$LOGS/nominal"
    CKPT_DIRS+=",rarl=$LOGS/adversarial"
    CKPT_DIRS+=",sa_mdp=$LOGS/sa_mdp"
    CKPT_DIRS+=",dr=$LOGS/domain_rand"
    CKPT_DIRS+=",rzsm=$LOGS/rzsm_best"

    $PY -m src.eval \
        --env "$ENV" \
        --methods "vanilla,rarl,sa_mdp,dr,rzsm" \
        --scenarios "nominal,force,params,noise,combined" \
        --checkpoint-dirs "$CKPT_DIRS" \
        --episodes $EVAL_EPISODES \
        --output "$RESULTS" \
        --device "$DEVICE" \
        --hidden-sizes "$BEST_HIDDEN" \
        --seq-len "$BEST_SEQ_LEN" \
        --d-model "$BEST_D_MODEL" \
        --transformer-layers "$BEST_LAYERS"
    done_ "Results saved → $RESULTS"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
H=$(( ELAPSED / 3600 ))
M=$(( (ELAPSED % 3600) / 60 ))
S=$(( ELAPSED % 60 ))

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Pipeline complete  (${H}h ${M}m ${S}s)${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo "  Checkpoints : $LOGS"
echo "  Results     : $RESULTS"
echo "  Tuning DB   : $TUNE_DB"
echo ""
echo "  Visualise Optuna:"
echo "    optuna-dashboard sqlite:///$TUNE_DB"
echo ""
