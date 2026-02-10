# Robust Zero-Sum Multi-Agent Reinforcement Learning

Zero-sum robust deep reinforcement learning with transformer-based disturbance detection and adaptive policy mixing for real-time control on embedded GPUs.

## Project Structure

```
.
├── src/                        # Source code
│   ├── agents/                 #   DDPG + Adversarial DDPG agents
│   ├── detector/               #   Transformer disturbance detector & blender
│   ├── networks/               #   MLP actor-critic architectures
│   ├── utils/                  #   Replay buffer, dataset logger
│   └── train.py                #   Training entry point
├── tests/                      # Unit tests
├── notebooks/                  # Experiment notebooks (Ant, Humanoid)
├── pretrained/                 # Pre-trained model weights & training logs
│   ├── ant/
│   └── humanoid/
├── report/                     # IEEE paper, references, and build artifacts
│   ├── paper.tex
│   ├── refs.bib
│   └── paper.pdf
├── requirements.txt
└── LICENSE
```

## Setup

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

## Architecture

- **Dual-policy**: `pi_opt` (nominal) + `pi_rob` (adversarial) with learned gating
- **Adversarial training**: Controller maximizes Q, adversary minimizes Q (gradient flip)
- **Transformer detector**: Lightweight encoder over observation history outputs disturbance probability and blending weight
- **Policy mixing**: `a_t = alpha * a_opt + (1 - alpha) * a_rob`

## Tests

```bash
python -m pytest tests/ -v
```

## License

See [LICENSE](LICENSE).
