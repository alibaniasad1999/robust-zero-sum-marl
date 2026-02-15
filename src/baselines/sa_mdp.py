"""
SA-MDP (State-Adversarial MDP) baseline.

Trains a standard TD3 policy with worst-case ℓ∞ bounded observation
perturbations applied during training.  At test time the policy is used
without perturbation — robustness comes from having trained against
adversarial state noise.

Reference: Zhang et al., "Robust Deep Reinforcement Learning against
Adversarial Perturbations on State Observations", NeurIPS 2020.

Usage:
    python -m src.baselines.sa_mdp --env Ant-v5 --epochs 100 --num-envs 10
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from src.agents.td3 import TD3Agent, _make_vec_env


class SAMDPAgent(TD3Agent):
    """TD3Agent with ℓ∞ state-adversarial perturbation during training.

    During the training loop, observations fed to the policy are perturbed
    by a worst-case noise bounded in ℓ∞ norm::

        obs_perturbed = obs + clip(uniform(-eps, eps), -eps, eps)

    The Q-function and replay buffer still receive the *true* observations
    so that value estimation remains unbiased.

    Args:
        eps: ℓ∞ perturbation radius (default 0.05).
        All other args are forwarded to ``TD3Agent``.
    """

    def __init__(self, env_fn, *, eps: float = 0.05, **kwargs):
        super().__init__(env_fn, **kwargs)
        self.eps = eps
        print(f"[SA-MDP] eps={eps}")

    def _perturb_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply ℓ∞ bounded uniform perturbation."""
        noise = np.random.uniform(-self.eps, self.eps, size=obs.shape).astype(obs.dtype)
        return obs + noise

    def train(self, epochs: Optional[int] = None) -> None:
        """SA-MDP training loop — same as TD3 but perturbs obs for action selection."""
        epochs = epochs or self.epochs
        N = self.num_envs

        # Overwrite CSV on each new training run
        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "episode_return"])

        total_transitions = self.steps_per_epoch * epochs
        total_steps = total_transitions // N

        obs, _ = self.envs.reset()
        ep_rets = np.zeros(N)
        ep_lens = np.zeros(N, dtype=int)
        t0 = time.time()

        for t in range(total_steps):
            global_step = t * N

            if global_step < self.start_steps:
                acts = np.stack([self.envs.single_action_space.sample()
                                 for _ in range(N)])
            else:
                # SA-MDP: perturb obs before policy forward pass
                obs_perturbed = self._perturb_obs(obs)
                acts = self._get_action(obs_perturbed, self.act_noise)

            next_obs, rews, terms, truncs, infos = self.envs.step(acts)

            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                for i in range(N):
                    if infos["_final_observation"][i]:
                        real_next_obs[i] = infos["final_observation"][i]

            # Store TRUE observations in buffer (not perturbed)
            self.buffer.add_batch(
                obs, real_next_obs, acts,
                rews.reshape(-1, 1),
                terms.reshape(-1, 1),
                truncs.reshape(-1, 1),
            )

            ep_rets += rews
            ep_lens += 1
            dones = terms | truncs
            for i in range(N):
                if dones[i]:
                    with open(self._csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([global_step, ep_rets[i]])
                    print(f"  step={global_step}  ret={ep_rets[i]:.1f}  "
                          f"len={ep_lens[i]}")
                    ep_rets[i] = 0.0
                    ep_lens[i] = 0

            obs = next_obs

            if global_step >= self.update_after and t % self.update_every == 0:
                for _ in range(self.n_updates):
                    batch = self.buffer.sample(self.batch_size)
                    self._update(batch)

            if (global_step + N) % self.steps_per_epoch < N:
                epoch = (global_step + N) // self.steps_per_epoch
                if epoch > 0:
                    print(f"[SA-MDP] Epoch {epoch}/{epochs}  "
                          f"elapsed={time.time()-t0:.0f}s")
                    if epoch % self.save_freq == 0:
                        self.save()


def main() -> None:
    p = argparse.ArgumentParser(description="Train SA-MDP baseline")
    p.add_argument("--env", type=str, default="Ant-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--start-steps", type=int, default=5000)
    p.add_argument("--update-after", type=int, default=10000)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--eps", type=float, default=0.05,
                    help="SA-MDP perturbation radius (ℓ∞)")
    args = p.parse_args()

    env_fn = lambda: gym.make(args.env)
    env_safe = args.env.replace("-", "_")
    log_dir = args.log_dir or f"logs/{env_safe}/sa_mdp"

    agent = SAMDPAgent(
        env_fn,
        eps=args.eps,
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
    agent.train()
    agent.save()
    print("Done.")


if __name__ == "__main__":
    main()
