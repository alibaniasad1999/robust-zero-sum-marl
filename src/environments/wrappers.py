"""
Gymnasium wrappers that inject test-time disturbances into MuJoCo envs.

Three disturbance types matching the paper's evaluation protocol:
    1. External impulse forces on the torso
    2. Physical parameter perturbation (mass, friction, damping)
    3. Additive Gaussian observation noise
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np


class ExternalForceWrapper(gym.Wrapper):
    """Apply random impulse forces to the torso body at random intervals.

    Uses MuJoCo's ``xfrc_applied`` field to inject a 6-DOF wrench
    (3 force + 3 torque) on the torso for a single timestep.

    Args:
        env: A MuJoCo Gymnasium environment.
        force_mag: Magnitude of the applied force (N).
        interval_range: (min, max) steps between consecutive impulses.
        torso_body_name: Name of the body to apply forces to.
    """

    def __init__(
        self,
        env: gym.Env,
        force_mag: float = 50.0,
        interval_range: Tuple[int, int] = (50, 200),
        torso_body_name: str = "torso",
    ):
        super().__init__(env)
        self.force_mag = force_mag
        self.interval_range = interval_range
        self.torso_body_name = torso_body_name
        self.disturbance_active = False
        self._steps_until_force = 0
        self._torso_id: int | None = None

    def _find_torso_id(self) -> int:
        model = self.unwrapped.model
        for i in range(model.nbody):
            name = model.body(i).name
            if name == self.torso_body_name:
                return i
        # Fallback: use body index 1 (first non-world body)
        return 1

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._torso_id = self._find_torso_id()
        self._steps_until_force = self.np_random.integers(
            self.interval_range[0], self.interval_range[1]
        )
        self.disturbance_active = False
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._steps_until_force -= 1
        data = self.unwrapped.data

        if self._steps_until_force <= 0:
            # Random 3D force direction
            direction = self.np_random.standard_normal(3)
            direction /= np.linalg.norm(direction) + 1e-8
            force = direction * self.force_mag
            # Apply as wrench: [fx, fy, fz, tx, ty, tz]
            data.xfrc_applied[self._torso_id, :3] = force
            data.xfrc_applied[self._torso_id, 3:] = 0.0
            self.disturbance_active = True
            self._steps_until_force = self.np_random.integers(
                self.interval_range[0], self.interval_range[1]
            )
        else:
            # Clear any previously applied force
            data.xfrc_applied[self._torso_id, :] = 0.0
            self.disturbance_active = False

        obs, rew, term, trunc, info = self.env.step(action)
        info["disturbance_active"] = self.disturbance_active
        return obs, rew, term, trunc, info


class ParameterPerturbationWrapper(gym.Wrapper):
    """Randomise physical parameters on each reset.

    Perturbs body masses, geom friction coefficients, and joint damping
    by a uniform factor in ``[1 - range, 1 + range]``.

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
        self.disturbance_active = True  # always active (param shift)
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

    def _perturb(self) -> None:
        model = self.unwrapped.model
        if not self._originals_saved:
            self._save_originals()

        # Restore originals first, then perturb
        model.body_mass[:] = self._orig_mass * (
            1.0 + self.np_random.uniform(-self.mass_range, self.mass_range,
                                         size=self._orig_mass.shape)
        )
        # Keep world body mass unchanged
        model.body_mass[0] = self._orig_mass[0]

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
        self._perturb()
        info["disturbance_active"] = True
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, rew, term, trunc, info = self.env.step(action)
        info["disturbance_active"] = True
        return obs, rew, term, trunc, info


class ObservationNoiseWrapper(gym.Wrapper):
    """Add Gaussian noise to observations.

    Args:
        env: A Gymnasium environment.
        noise_std: Standard deviation of the additive noise.
    """

    def __init__(self, env: gym.Env, noise_std: float = 0.05):
        super().__init__(env)
        self.noise_std = noise_std
        self.disturbance_active = True

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        obs = obs + self.np_random.standard_normal(obs.shape).astype(
            obs.dtype) * self.noise_std
        info["disturbance_active"] = True
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, rew, term, trunc, info = self.env.step(action)
        obs = obs + self.np_random.standard_normal(obs.shape).astype(
            obs.dtype) * self.noise_std
        info["disturbance_active"] = True
        return obs, rew, term, trunc, info


class CombinedDisturbanceWrapper(gym.Wrapper):
    """Apply all three disturbance types simultaneously.

    Composes ExternalForceWrapper, ParameterPerturbationWrapper, and
    ObservationNoiseWrapper.

    Args:
        env: A MuJoCo Gymnasium environment.
        force_mag: Force magnitude for impulse disturbances.
        mass_range: Fractional range for mass perturbation.
        friction_range: Fractional range for friction perturbation.
        damping_range: Fractional range for damping perturbation.
        noise_std: Observation noise standard deviation.
    """

    def __init__(
        self,
        env: gym.Env,
        force_mag: float = 50.0,
        mass_range: float = 0.3,
        friction_range: float = 0.3,
        damping_range: float = 0.3,
        noise_std: float = 0.05,
    ):
        env = ParameterPerturbationWrapper(env, mass_range, friction_range, damping_range)
        env = ExternalForceWrapper(env, force_mag)
        env = ObservationNoiseWrapper(env, noise_std)
        super().__init__(env)
        self.disturbance_active = True
