"""
Domain Randomization (DR) baseline.

Trains a standard TD3 policy on environments with randomised physical
parameters (mass, friction, damping) each episode.  No adversary, no
transformer detector — robustness comes purely from seeing diverse dynamics.

Reference: Tobin et al., "Domain Randomization for Transferring Deep Neural
Networks from Simulation to the Real World", IROS 2017.

Usage:
    python -m src.baselines.domain_randomization --env Ant-v5 --epochs 100 --num-envs 10
"""

from __future__ import annotations

import argparse
from typing import Any, Tuple

import gymnasium as gym
import numpy as np

from src.agents.td3 import TD3Agent


class DomainRandomizationWrapper(gym.Wrapper):
    """Randomise MuJoCo physical parameters on each episode reset.

    Same perturbation logic as ``ParameterPerturbationWrapper`` but intended
    for *training* (the policy learns to be robust across dynamics).

    Args:
        env: A MuJoCo Gymnasium environment.
        mass_range: Fractional range for body mass perturbation.
        friction_range: Fractional range for geom friction perturbation.
        damping_range: Fractional range for joint damping perturbation.
    """

    def __init__(
        self,
        env: gym.Env,
        mass_range: float = 0.3,
        friction_range: float = 0.3,
        damping_range: float = 0.3,
    ):
        super().__init__(env)
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.damping_range = damping_range
        self._originals_saved = False
        self._orig_mass: np.ndarray | None = None
        self._orig_friction: np.ndarray | None = None
        self._orig_damping: np.ndarray | None = None

    def _save_originals(self) -> None:
        model = self.unwrapped.model
        self._orig_mass = model.body_mass.copy()
        self._orig_friction = model.geom_friction.copy()
        self._orig_damping = model.dof_damping.copy()
        self._originals_saved = True

    def _randomise(self) -> None:
        model = self.unwrapped.model
        if not self._originals_saved:
            self._save_originals()

        model.body_mass[:] = self._orig_mass * (
            1.0 + self.np_random.uniform(-self.mass_range, self.mass_range,
                                         size=self._orig_mass.shape)
        )
        model.body_mass[0] = self._orig_mass[0]  # keep world body

        model.geom_friction[:] = self._orig_friction * (
            1.0 + self.np_random.uniform(-self.friction_range, self.friction_range,
                                         size=self._orig_friction.shape)
        )
        model.dof_damping[:] = self._orig_damping * (
            1.0 + self.np_random.uniform(-self.damping_range, self.damping_range,
                                         size=self._orig_damping.shape)
        )

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._randomise()
        return obs, info


def main() -> None:
    p = argparse.ArgumentParser(description="Train Domain Randomization baseline")
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
    p.add_argument("--mass-range", type=float, default=0.3)
    p.add_argument("--friction-range", type=float, default=0.3)
    p.add_argument("--damping-range", type=float, default=0.3)
    args = p.parse_args()

    mr, fr, dr_ = args.mass_range, args.friction_range, args.damping_range
    env_safe = args.env.replace("-", "_")
    log_dir = args.log_dir or f"logs/{env_safe}/dr"

    def env_fn():
        return DomainRandomizationWrapper(
            gym.make(args.env),
            mass_range=mr,
            friction_range=fr,
            damping_range=dr_,
        )

    agent = TD3Agent(
        env_fn,
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
