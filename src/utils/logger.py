"""
Dataset logger matching the paper's per-timestep data logging specification
(Table I) and sequence window extraction for transformer training.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TransitionRecord:
    """Single timestep record matching the paper's logging spec."""

    t: int
    episode_id: int
    obs: np.ndarray
    action_ctrl: np.ndarray
    action_dist: np.ndarray
    action_total: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    disturbance_params: np.ndarray
    disturbance_tag: str = "none"
    robust_weight_target: float = 0.0


class DatasetLogger:
    """
    Logs every transition to disk for offline transformer training.

    Storage format: numpy .npz archive per episode (fast, no extra deps).
    Also writes a summary CSV for quick inspection.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._episode_buffer: List[TransitionRecord] = []
        self._episode_id: int = 0
        self._csv_path = os.path.join(log_dir, "transitions.csv")
        self._csv_header_written = os.path.isfile(self._csv_path)

    def log(self, record: TransitionRecord) -> None:
        self._episode_buffer.append(record)

    def end_episode(self) -> None:
        """Flush the current episode buffer to disk as a .npz file."""
        if not self._episode_buffer:
            return

        ep = self._episode_buffer
        n = len(ep)
        obs = np.array([r.obs for r in ep], dtype=np.float32)
        act_ctrl = np.array([r.action_ctrl for r in ep], dtype=np.float32)
        act_dist = np.array([r.action_dist for r in ep], dtype=np.float32)
        act_total = np.array([r.action_total for r in ep], dtype=np.float32)
        rewards = np.array([r.reward for r in ep], dtype=np.float32)
        terminated = np.array([r.terminated for r in ep], dtype=np.bool_)
        truncated = np.array([r.truncated for r in ep], dtype=np.bool_)
        dist_params = np.array([r.disturbance_params for r in ep], dtype=np.float32)
        dist_tags = np.array([r.disturbance_tag for r in ep])
        alpha_targets = np.array(
            [r.robust_weight_target for r in ep], dtype=np.float32
        )

        episode_dir = os.path.join(self.log_dir, "episodes")
        os.makedirs(episode_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(episode_dir, f"ep_{self._episode_id:06d}.npz"),
            obs=obs,
            action_ctrl=act_ctrl,
            action_dist=act_dist,
            action_total=act_total,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            disturbance_params=dist_params,
            disturbance_tag=dist_tags,
            robust_weight_target=alpha_targets,
            episode_id=np.array(self._episode_id),
        )

        # append summary row to CSV
        with open(self._csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not self._csv_header_written:
                w.writerow(
                    [
                        "episode_id",
                        "length",
                        "total_reward",
                        "has_disturbance",
                        "mean_alpha_target",
                    ]
                )
                self._csv_header_written = True
            has_dist = any(r.disturbance_tag != "none" for r in ep)
            w.writerow(
                [
                    self._episode_id,
                    n,
                    float(rewards.sum()),
                    int(has_dist),
                    float(alpha_targets.mean()),
                ]
            )

        self._episode_id += 1
        self._episode_buffer.clear()

    # ------------------------------------------------------------------
    # Sequence extraction for transformer training
    # ------------------------------------------------------------------
    @staticmethod
    def extract_windows(
        episode_dir: str,
        seq_length: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract sliding windows from all logged episodes.

        Returns:
            obs_windows:  (N, seq_length, obs_dim)
            dist_labels:  (N,)   0/1 disturbance present
            alpha_labels: (N,)   target blending weight
        """
        import glob

        files = sorted(glob.glob(os.path.join(episode_dir, "ep_*.npz")))
        all_obs, all_dist, all_alpha = [], [], []

        for path in files:
            data = np.load(path, allow_pickle=True)
            obs = data["obs"]  # (T, obs_dim)
            tags = data["disturbance_tag"]
            alphas = data["robust_weight_target"]
            T = obs.shape[0]

            for i in range(T):
                start = max(0, i - seq_length + 1)
                window = obs[start : i + 1]
                # zero-pad if window is shorter than seq_length
                if window.shape[0] < seq_length:
                    pad = np.zeros(
                        (seq_length - window.shape[0], obs.shape[1]),
                        dtype=np.float32,
                    )
                    window = np.concatenate([pad, window], axis=0)
                all_obs.append(window)
                all_dist.append(float(tags[i] != "none"))
                all_alpha.append(float(alphas[i]))

        return (
            np.array(all_obs, dtype=np.float32),
            np.array(all_dist, dtype=np.float32),
            np.array(all_alpha, dtype=np.float32),
        )
