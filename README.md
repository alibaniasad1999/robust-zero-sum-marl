# Robust Zero-Sum MARL (RZSM)

Zero-sum robust deep reinforcement learning with transformer-based disturbance detection and adaptive policy mixing. Includes Python training, a full pipeline runner, C++ native port (LibTorch + MuJoCo), and ROS 2 deployment for NVIDIA Jetson.

## Project Structure

```
.
├── src/                          # Python source
│   ├── agents/td3.py             #   TD3Agent (nominal) + AdversarialTD3Agent (zero-sum)
│   ├── detector/transformer.py   #   Transformer detector, gating network, blender
│   ├── networks/mlp.py           #   MLP actor-critic architectures
│   ├── baselines/                #   SA-MDP + Domain Randomisation baselines
│   ├── environments/wrappers.py  #   Disturbance wrappers (force / param / noise)
│   ├── utils/
│   │   ├── logger.py             #     DatasetLogger — per-step .npz episode archives
│   │   ├── perf_logger.py        #     GPU/CPU throughput logging
│   │   └── buffers/              #     Vectorised replay buffer
│   ├── train.py                  #   Single-run training entry point
│   └── eval.py                   #   Multi-scenario evaluation
├── scripts/
│   ├── run_pipeline.sh           #   *** Full end-to-end pipeline (see below) ***
│   ├── tune_rzsm.py              #   Optuna hyperparameter tuning (two-phase)
│   ├── train_transformer.py      #   Offline transformer training on logged data
│   ├── plot_results.py           #   Result plotting
│   └── analyze_rzsm.py           #   Analysis utilities
├── cpp/                          # C++ port (LibTorch + MuJoCo C API)
│   ├── CMakeLists.txt
│   ├── include/rzsm/
│   ├── src/
│   ├── tests/                    #   64 GoogleTest unit tests
│   └── assets/                   #   MuJoCo XML models (ant, humanoid)
├── ros2_ws/                      # ROS 2 workspace (Jetson deployment)
│   └── src/rzsm_control/
│       ├── src/                  #   Controller node (200-500 Hz)
│       ├── msg/                  #   DetectorStatus custom message
│       ├── config/               #   YAML parameters
│       ├── launch/               #   Launch files
│       └── scripts/              #   ONNX export utility
├── tests/                        # Python unit tests (73 tests)
├── notebooks/                    # Experiment notebooks (Ant, Humanoid)
├── pretrained/                   # Pre-trained model weights
├── report/                       # IEEE paper, references, LaTeX source
├── requirements.txt
└── LICENSE
```

---

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Training and ONNX export |
| PyTorch | 2.0+ | Neural networks |
| MuJoCo | 3.x | Physics simulation |
| C++ Compiler | C++17 | GCC 9+ / Clang 14+ |
| CMake | 3.18+ | Build system |
| LibTorch | Matching PyTorch | C++ frontend |
| ROS 2 | Jazzy / Humble | Jetson deployment (optional) |
| JetPack | 6.x | NVIDIA Jetson (optional) |

## Python Setup

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

---

## Full Pipeline (Recommended)

`scripts/run_pipeline.sh` runs the **complete end-to-end training sequence** automatically:

1. **Baselines** — SA-MDP and Domain Randomisation (data collection + logging)
2. **Nominal TD3** — trains `pi_opt`, logs every transition to `dataset/`
3. **Adversarial TD3** — trains `pi_rob + pi_adv` with epsilon annealing, logs data
4. **Offline Transformer** — trains detector on the aggregated dataset from all runs
5. **Optuna Tuning** — two-phase Bayesian hyperparameter search for RZSM
6. **RZSM Final** — trains RZSM with best hyperparameters (1M steps)
7. **Evaluation** — all methods across 5 disturbance scenarios

```bash
# Default: runs Ant-v5 with sensible defaults
bash scripts/run_pipeline.sh

# Select a different environment
bash scripts/run_pipeline.sh --env HalfCheetah-v5

# Skip specific stages (useful for resuming)
bash scripts/run_pipeline.sh --env Ant-v5 --skip-baselines --skip-nominal

# All flags
bash scripts/run_pipeline.sh --help
```

### Pipeline Stages

| Stage | Flag to skip | Output |
|-------|-------------|--------|
| SA-MDP baseline | `--skip-baselines` | `logs/{env}/sa_mdp/` |
| Domain Randomisation baseline | `--skip-baselines` | `logs/{env}/domain_rand/` |
| Nominal TD3 (pi_opt) | `--skip-nominal` | `logs/{env}/nominal/seed_0/` |
| Adversarial TD3 (pi_rob + pi_adv) | `--skip-adversarial` | `logs/{env}/adversarial/seed_0/` |
| Offline transformer training | `--skip-transformer` | `logs/{env}/detector_offline.pt` |
| Optuna hyperparameter tuning | `--skip-tune` | `logs/{env}/tuning/optuna.db` |
| RZSM final training (best params) | `--skip-rzsm` | `logs/{env}/rzsm_best/` |
| Evaluation (all methods) | `--skip-eval` | `results/{env}/` |

---

## Manual Training

### Phase 1 — Nominal policy (pi_opt)

```bash
python -m src.train --env Ant-v5 --mode nominal --epochs 200 --seed 0
```

### Phase 2 — Adversarial zero-sum game (pi_rob + pi_adv)

```bash
python -m src.train --env Ant-v5 --mode adversarial --epochs 200 \
    --pi-opt-path logs/Ant_v5/nominal/seed_0/checkpoints \
    --warmup-fraction 0.2
```

### Phase 3 — Offline transformer training

```bash
python scripts/train_transformer.py \
    --log-dirs \
        logs/Ant_v5/nominal/seed_0/dataset \
        logs/Ant_v5/adversarial/seed_0/dataset \
    --obs-dim 27 \
    --output logs/Ant_v5/detector_offline.pt \
    --epochs 50 --batch-size 512
```

### Phase 4 — RZSM fine-tuning with pre-trained detector

```bash
# Phase 1 fine-tune: learn blending with frozen detector
python -m src.train --env Ant-v5 --mode adversarial --epochs 100 \
    --pi-opt-path logs/Ant_v5/nominal/seed_0/checkpoints \
    --log-dir logs/Ant_v5/rzsm_phase1/seed_0

# Phase 2 fine-tune: extended training for best results
python -m src.train --env Ant-v5 --mode adversarial --epochs 200 \
    --pi-opt-path logs/Ant_v5/nominal/seed_0/checkpoints \
    --log-dir logs/Ant_v5/rzsm_phase2/seed_0 \
    --disturbance-ratio 0.05 --warmup-fraction 0.1
```

### Key Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `Ant-v5` | Gymnasium environment ID |
| `--mode` | `nominal` | `nominal` or `adversarial` |
| `--epochs` | `200` | Training epochs |
| `--steps-per-epoch` | `4000` | Environment steps per epoch |
| `--batch-size` | `256` | Replay buffer sample size |
| `--num-envs` | `1` | Parallel environments (vectorised) |
| `--disturbance-ratio` | `0.05` | Max adversary action scale |
| `--disturbance-prob` | `0.3` | Probability of disturbance per episode |
| `--warmup-fraction` | `0.2` | Fraction of steps for epsilon annealing (0→ε_max) |
| `--pi-opt-path` | `None` | Path to pre-trained nominal policy checkpoints |
| `--no-transformer` | off | Disable transformer detector |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

---

## Architecture

### Training Flow

```
TD3Agent (nominal)          →  pi_opt
AdversarialTD3Agent         →  pi_rob + pi_adv  (zero-sum, gradient flip)
TransformerDisturbanceDetector  →  p_t, δ̂_t, α_t
AdaptiveControllerBlender   →  a_t = α_t · a_opt + (1 − α_t) · a_rob
```

### Gating Network (Paper Eq 15)

```
obs_history  →  TransformerEncoder  →  final token embedding
                   ├── disturbance_head  →  p_t  ∈ [0,1]
                   ├── magnitude_head    →  δ̂_t  (L2 norm estimate)
                   └── blending_head([p_t, δ̂_t])  →  α_t  ∈ [0,1]
```

The gating head `f_φ` receives `[p_t, δ̂_t]` as its input (not the raw embedding), matching the paper exactly.

### Adversarial Curriculum

Disturbance budget is linearly annealed: ε(t) = ε_max · min(1, t / t_warmup), where t_warmup = 20% of total training steps by default.

### Dataset Logging

Every transition from every environment and every training mode is saved to `logs/{env}/{mode}/seed_0/dataset/episodes/ep_XXXXXX.npz`. This dataset is aggregated by `scripts/train_transformer.py` for large-scale offline training of the detector.

---

## Evaluation

```bash
python -m src.eval \
    --env Ant-v5 \
    --methods vanilla,rarl,rzsm \
    --scenarios nominal,force,params,noise,combined \
    --checkpoint-dir logs/ \
    --episodes 50 \
    --output results/
```

---

## C++ Build

The C++ port mirrors the full Python framework using LibTorch and the MuJoCo C API.

```bash
cd cpp && mkdir build && cd build

# Linux + CUDA
cmake .. \
    -DCMAKE_PREFIX_PATH="/path/to/libtorch" \
    -DMUJOCO_DIR="/path/to/mujoco" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# macOS (CPU only)
cmake .. \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -DMUJOCO_DIR="/opt/homebrew/opt/mujoco"
cmake --build . --parallel

# Run C++ tests
ctest --output-on-failure

# C++ training
./train --env Ant --mode nominal --epochs 100 --device auto
./train --env Ant --mode adversarial --epochs 100 \
    --pi-opt-path logs/nominal_Ant/checkpoints
```

---

## ROS 2 Deployment (Jetson)

```bash
# 1. Build C++ library first
cd cpp && mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch" -DMUJOCO_DIR="/path/to/mujoco"
cmake --build . --parallel && cd ../..

# 2. Build ROS 2 workspace
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --cmake-args \
    -DCMAKE_PREFIX_PATH="/path/to/libtorch" \
    -DRZSM_LIB_DIR=$(realpath ../cpp/build)
source install/setup.bash

# 3. Launch
ros2 launch rzsm_control robust_controller.launch.py

# Monitor
ros2 topic echo /detector_status
ros2 topic hz /cmd_vel
```

### ROS 2 Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Sub | Joint positions/velocities |
| `/imu` | `sensor_msgs/Imu` | Sub | IMU sensor data |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | Pub | Blended control command |
| `/detector_status` | `rzsm_control/DetectorStatus` | Pub | p_t, α_t, latency |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Pub | Health/timing |

---

## ONNX Export

```bash
python3 ros2_ws/src/rzsm_control/scripts/export_onnx.py \
    --pi-opt pretrained/ant/model/actor_cuda.pth \
    --pi-rob pretrained/ant/model/actor_rob_cuda.pth \
    --detector pretrained/ant/model/disturbance_cuda.pth \
    --obs-dim 27 --act-dim 8 \
    --output-dir models/onnx/

# TensorRT on Jetson
trtexec --onnx=models/onnx/pi_opt.onnx --saveEngine=pi_opt.engine --fp16
```

---

## Tests

```bash
# Python (73 tests)
python -m pytest tests/ -v

# C++ (64 tests)
cd cpp/build && ctest --output-on-failure
```

---

## License

See [LICENSE](LICENSE).
