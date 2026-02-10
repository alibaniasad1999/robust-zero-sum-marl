#!/usr/bin/env python3
"""
Entry-point for training DDPG agents.

Usage:
    # Phase 1 — train optimal policy on nominal environment
    python -m src.train --env Ant-v5 --mode nominal --epochs 100

    # Phase 2 — train robust policy + adversary in zero-sum game
    python -m src.train --env Ant-v5 --mode adversarial --epochs 100 \
        --pi-opt-path logs/nominal/checkpoints
"""

import argparse

import gymnasium as gym

from src.agents.ddpg import DDPGAgent, AdversarialDDPGAgent


def make_env(env_name: str):
    def _thunk():
        return gym.make(env_name)
    return _thunk


def main() -> None:
    p = argparse.ArgumentParser(description="Train DDPG agents")
    p.add_argument("--env", type=str, default="Ant-v5", help="Gymnasium env id")
    p.add_argument("--mode", choices=["nominal", "adversarial"], default="nominal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--start-steps", type=int, default=5000)
    p.add_argument("--update-after", type=int, default=10000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log-dir", type=str, default=None)
    # adversarial-specific
    p.add_argument("--disturbance-ratio", type=float, default=0.1)
    p.add_argument("--disturbance-prob", type=float, default=0.5)
    p.add_argument("--pi-opt-path", type=str, default=None,
                    help="Path to pre-trained optimal policy checkpoints (for blending)")
    p.add_argument("--no-transformer", action="store_true",
                    help="Disable transformer disturbance detector")
    args = p.parse_args()

    env_fn = make_env(args.env)
    log_dir = args.log_dir or f"logs/{args.mode}"

    if args.mode == "nominal":
        agent = DDPGAgent(
            env_fn,
            seed=args.seed,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            update_after=args.update_after,
            device=args.device,
            log_dir=log_dir,
        )
    else:
        agent = AdversarialDDPGAgent(
            env_fn,
            seed=args.seed,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            update_after=args.update_after,
            device=args.device,
            log_dir=log_dir,
            disturbance_ratio=args.disturbance_ratio,
            disturbance_probability=args.disturbance_prob,
            use_transformer=not args.no_transformer,
            pi_opt_path=args.pi_opt_path,
        )

    agent.train()
    agent.save()
    print("Done.")


if __name__ == "__main__":
    main()
