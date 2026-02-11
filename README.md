# Robust Zero-Sum Multi-Agent Reinforcement Learning

Zero-sum robust deep reinforcement learning with transformer-based disturbance detection and adaptive policy mixing. Includes Python training, C++ native port (LibTorch + MuJoCo), and ROS 2 deployment for NVIDIA Jetson.

## Project Structure

```
.
├── src/                          # Python source code
│   ├── agents/                   #   DDPG + Adversarial DDPG agents
│   ├── detector/                 #   Transformer disturbance detector & blender
│   ├── networks/                 #   MLP actor-critic architectures
│   ├── utils/                    #   Replay buffer, dataset logger
│   └── train.py                  #   Training entry point
├── cpp/                          # C++ port (LibTorch + MuJoCo C API)
│   ├── CMakeLists.txt
│   ├── include/rzsm/             #   Headers (namespace rzsm)
│   ├── src/                      #   Implementation + main.cpp CLI
│   ├── tests/                    #   GoogleTest unit tests (64 tests)
│   └── assets/                   #   MuJoCo XML models (ant, humanoid)
├── ros2_ws/                      # ROS 2 workspace (Jetson deployment)
│   └── src/rzsm_control/         #   ROS 2 control package
│       ├── src/                  #     Controller node (200-500 Hz)
│       ├── msg/                  #     DetectorStatus custom message
│       ├── config/               #     YAML parameters
│       ├── launch/               #     Launch files
│       └── scripts/              #     ONNX export utility
├── tests/                        # Python unit tests (73 tests)
├── notebooks/                    # Experiment notebooks (Ant, Humanoid)
├── pretrained/                   # Pre-trained model weights
├── report/                       # IEEE paper, references, LaTeX source
├── requirements.txt
└── LICENSE
```

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Training and ONNX export |
| PyTorch | 2.0+ | Neural networks |
| MuJoCo | 3.x | Physics simulation |
| C++ Compiler | C++17 | GCC 9+ / Clang 14+ |
| CMake | 3.18+ | Build system |
| LibTorch | Matching PyTorch | C++ frontend (auto-downloaded or manual) |
| ROS 2 | Jazzy / Humble | Jetson deployment (optional) |
| JetPack | 6.x | NVIDIA Jetson (optional) |

## Python Setup

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Training

**Phase 1** -- Train optimal policy on nominal environment:
```bash
python -m src.train --env Ant-v5 --mode nominal --epochs 100
```

**Phase 2** -- Train robust policy + adversary in zero-sum game:
```bash
python -m src.train --env Ant-v5 --mode adversarial --epochs 100 \
    --pi-opt-path logs/nominal/checkpoints
```

### Key Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `Ant-v5` | Gymnasium environment ID |
| `--mode` | `nominal` | `nominal` or `adversarial` |
| `--epochs` | `100` | Number of training epochs |
| `--steps-per-epoch` | `4000` | Environment steps per epoch |
| `--batch-size` | `1024` | Replay buffer sample size |
| `--disturbance-ratio` | `0.1` | Adversary action scale (adversarial mode) |
| `--disturbance-prob` | `0.5` | Probability of applying disturbance |
| `--no-transformer` | `false` | Disable transformer detector |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

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

# Run tests
ctest --output-on-failure
```

### C++ Training

```bash
./train --env Ant --mode nominal --epochs 100 --device auto
./train --env Ant --mode adversarial --epochs 100 \
    --pi-opt-path logs/nominal_Ant/checkpoints
./train --help   # all options
```

## ROS 2 Deployment (Jetson)

The `ros2_ws/` package deploys trained models on NVIDIA Jetson at 200-500 Hz.

### Build

```bash
# 1. Build the C++ library first
cd cpp && mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch" -DMUJOCO_DIR="/path/to/mujoco"
cmake --build . --parallel
cd ../..

# 2. Build the ROS 2 workspace
source /opt/ros/jazzy/setup.bash   # or humble
cd ros2_ws
colcon build --cmake-args \
    -DCMAKE_PREFIX_PATH="/path/to/libtorch" \
    -DRZSM_LIB_DIR=$(realpath ../cpp/build)
source install/setup.bash
```

### Run

```bash
# Launch with default parameters (Ant, 200 Hz)
ros2 launch rzsm_control robust_controller.launch.py

# Custom parameters
ros2 launch rzsm_control robust_controller.launch.py \
    params_file:=/path/to/custom_params.yaml

# Monitor
ros2 topic echo /detector_status    # disturbance prob, alpha, latency
ros2 topic hz /cmd_vel              # verify control frequency
```

### Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Subscribe | Robot joint positions/velocities |
| `/imu` | `sensor_msgs/Imu` | Subscribe | IMU sensor data |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | Publish | Blended control command |
| `/detector_status` | `rzsm_control/DetectorStatus` | Publish | Disturbance prob, alpha, latency |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Publish | Health and timing diagnostics |

## ONNX Export

Export trained models for TensorRT optimization on Jetson:

```bash
python3 ros2_ws/src/rzsm_control/scripts/export_onnx.py \
    --pi-opt pretrained/ant/model/actor_cuda.pth \
    --pi-rob pretrained/ant/model/actor_rob_cuda.pth \
    --detector pretrained/ant/model/disturbance_cuda.pth \
    --obs-dim 27 --act-dim 8 \
    --output-dir models/onnx/

# Then on Jetson:
trtexec --onnx=models/onnx/pi_opt.onnx --saveEngine=pi_opt.engine --fp16
```

## Architecture

- **Dual-policy**: `pi_opt` (nominal) + `pi_rob` (adversarial) with learned gating
- **Adversarial training**: Controller maximizes Q, adversary minimizes Q (gradient flip)
- **Transformer detector**: Lightweight encoder over observation history outputs disturbance probability and blending weight
- **Policy mixing**: `a_t = alpha * a_opt + (1 - alpha) * a_rob`
- **C++ port**: Namespace `rzsm`, LibTorch modules, MuJoCo C API (Ant + Humanoid)
- **ROS 2 node**: 200 Hz timer callback, mutex-protected sensor fusion, pre-allocated GPU tensors

## Tests

```bash
# Python (73 tests)
python -m pytest tests/ -v

# C++ (64 tests, requires build)
cd cpp/build && ctest --output-on-failure
```

## License

See [LICENSE](LICENSE).
