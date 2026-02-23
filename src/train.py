#!/usr/bin/env python3
"""
Entry-point for training TD3 agents.

Usage:
    # Phase 1 — train optimal policy on nominal environment
    python -m src.train --env Ant-v5 --mode nominal --epochs 100 --seed 0

    # Phase 1 with 10 parallel envs (Colab / multi-core)
    python -m src.train --env Ant-v5 --mode nominal --epochs 100 --num-envs 10

    # Phase 2 — train robust policy + adversary in zero-sum game
    python -m src.train --env Ant-v5 --mode adversarial --epochs 100 \
        --pi-opt-path logs/nominal/seed_0/checkpoints

    # Multi-seed training (run each seed separately)
    for seed in 0 1 2 3 4; do
        python -m src.train --env Ant-v5 --mode nominal --epochs 100 --seed $seed
    done
    # Saves to: logs/nominal/seed_0/, logs/nominal/seed_1/, ...
"""

import argparse
import os

import gymnasium as gym

from src.agents.td3 import TD3Agent, AdversarialTD3Agent


def make_env(env_name: str):
    def _thunk():
        return gym.make(env_name)
    return _thunk


def main() -> None:
    p = argparse.ArgumentParser(description="Train TD3 agents")
    p.add_argument("--env", type=str, default="Ant-v5", help="Gymnasium env id")
    p.add_argument("--mode", choices=["nominal", "adversarial"], default="nominal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--start-steps", type=int, default=10000)
    p.add_argument("--update-after", type=int, default=10000)
    p.add_argument("--num-envs", type=int, default=1,
                    help="Number of parallel environments (vectorized)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log-dir", type=str, default=None)
    # adversarial-specific
    p.add_argument("--disturbance-ratio", type=float, default=0.05)
    p.add_argument("--disturbance-prob", type=float, default=0.3)
    p.add_argument("--warmup-fraction", type=float, default=0.2,
                    help="Fraction of training steps to linearly anneal disturbance budget (default: 0.2)")
    p.add_argument("--pi-opt-path", type=str, default=None,
                    help="Path to pre-trained optimal policy checkpoints (for blending)")
    p.add_argument("--no-transformer", action="store_true",
                    help="Disable transformer disturbance detector")
    # network architecture
    p.add_argument("--hidden-sizes", type=str, default="256,256",
                    help="Comma-separated MLP hidden layer sizes (default: 256,256)")
    # transformer detector hyperparameters
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=3)
    p.add_argument("--detector-lr", type=float, default=1e-4)
    p.add_argument("--detector-train-interval", type=int, default=200)
    args = p.parse_args()
    args.hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(","))

    env_fn = make_env(args.env)
    env_safe = args.env.replace("-", "_")
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(f"logs/{env_safe}/{args.mode}", f"seed_{args.seed}")

    if args.mode == "nominal":
        agent = TD3Agent(
            env_fn,
            hidden_sizes=args.hidden_sizes,
            seed=args.seed,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            update_after=args.update_after,
            num_envs=args.num_envs,
            device=args.device,
            log_dir=log_dir,
        )
    else:
        agent = AdversarialTD3Agent(
            env_fn,
            hidden_sizes=args.hidden_sizes,
            seed=args.seed,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            update_after=args.update_after,
            num_envs=args.num_envs,
            device=args.device,
            log_dir=log_dir,
            disturbance_ratio=args.disturbance_ratio,
            disturbance_probability=args.disturbance_prob,
            warmup_fraction=args.warmup_fraction,
            use_transformer=not args.no_transformer,
            pi_opt_path=args.pi_opt_path,
            transformer_seq_len=args.seq_len,
            transformer_d_model=args.d_model,
            transformer_nhead=args.nhead,
            transformer_layers=args.transformer_layers,
            transformer_lr=args.detector_lr,
            transformer_train_interval=args.detector_train_interval,
        )

    agent.train()
    agent.save()
    print("Done.")


if __name__ == "__main__":
    main()
