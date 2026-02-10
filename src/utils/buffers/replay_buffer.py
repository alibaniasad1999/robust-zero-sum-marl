"""
    Buffer for Reinforcement Learning algorythms.
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import torch
import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    obs: torch.Tensor         # [B, obs_dim]
    next_obs: torch.Tensor    # [B, obs_dim]
    act: torch.Tensor         # [B, act_dim_total] (concatenated for multi-agent)
    rew: torch.Tensor         # [B, 1]
    terminated: torch.Tensor  # [B, 1]
    truncated: torch.Tensor   # [B, 1]

    @property
    def done(self) -> torch.Tensor:
        # [B, 1]
        return self.terminated | self.truncated


class ReplayBuffer():
    def __init__(self,
                 capacity: int,
                 obs_dim: int,
                 total_act: int,
                 device: torch.device,
                 dtype: np.dtype,
                 env_num: Optional[int] = None) -> None:

        assert capacity > 0, "Capacity should be more that zero."

        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.total_act = total_act
        self.device = device
        self.dtype = dtype
        self.env_num = env_num

        self.obs_buffer = np.zeros((self.capacity, self.obs_dim), dtype=self.dtype)
        self.next_obs_buffer = np.zeros((self.capacity, self.obs_dim), dtype=self.dtype)
        self.act_buffer = np.zeros((self.capacity, self.total_act), dtype=self.dtype)
        self.rew_buffer = np.zeros((self.capacity, 1), dtype=self.dtype)
        self.terminated_buffer = np.zeros((self.capacity, 1), dtype=np.bool_)
        self.truncated_buffer = np.zeros((self.capacity, 1), dtype=np.bool_)

        self.ptr: int = 0
        self.size: int = 0

    def add(
        self,
        obs: np.ndarray,        # shape (obs_dim,)
        next_obs: np.ndarray,   # shape (obs_dim,)
        act: np.ndarray,        # shape (total_act,)
        rew: float,             # scalar
        terminated: bool,       # scalar
        truncated: bool,        # scalar
    ) -> None:
        obs_b = np.asarray(obs, dtype=self.dtype).reshape(1, self.obs_dim)
        next_obs_b = np.asarray(next_obs, dtype=self.dtype).reshape(1, self.obs_dim)
        act_b = np.asarray(act, dtype=self.dtype).reshape(1, self.total_act)
        rew_b = np.asarray(rew, dtype=self.dtype).reshape(1, 1)
        terminated_b = np.asarray(terminated, dtype=np.bool_).reshape(1, 1)
        truncated_b = np.asarray(truncated, dtype=np.bool_).reshape(1, 1)

        self.add_batch(obs_b, next_obs_b, act_b, rew_b, terminated_b, truncated_b)

    def add_batch(
        self,
        obs: np.ndarray,         # shape (N, obs_dim)
        next_obs: np.ndarray,    # shape (N, obs_dim)
        act: np.ndarray,         # shape (N, total_dim)
        rew: np.ndarray,         # shape (N, 1)
        terminated: np.ndarray,  # shape (N, 1)
        truncated: np.ndarray    # shape (N, 1)
    ) -> None:
        """
        Add a batch of transitions from a vectorized environment.

        Args:
            obs: Observations, shape (N, obs_dim).
            next_obs: Next observations, shape (N, obs_dim).
            act: Actions, shape (N, act_dim_total).
            rew: Rewards, shape (N,) or (N, 1).
            terminated: Terminated flags, shape (N,) or (N, 1).
            truncated: Truncated flags, shape (N,) or (N, 1).
        """
        N = int(obs.shape[0])
        if self.env_num is not None:
            assert N == self.env_num, f"Expected N={self.env_num}, got N={N}"
        obs = np.asarray(obs, dtype=self.dtype).reshape(N, self.obs_dim)
        next_obs = np.asarray(next_obs, dtype=self.dtype).reshape(N, self.obs_dim)
        act = np.asarray(act, dtype=self.dtype).reshape(N, self.total_act)
        rew = np.asarray(rew, dtype=self.dtype).reshape(N, 1)
        terminated = np.asarray(terminated, dtype=np.bool_).reshape(N, 1)
        truncated = np.asarray(truncated, dtype=np.bool_).reshape(N, 1)

        p = self.ptr
        end = p + N

        if end <= self.capacity:
            sl = slice(p, end)
            self.obs_buffer[sl, :] = obs
            self.next_obs_buffer[sl, :] = next_obs
            self.act_buffer[sl, :] = act
            self.rew_buffer[sl, :] = rew
            self.terminated_buffer[sl, :] = terminated
            self.truncated_buffer[sl, :] = truncated
        else:
            k = self.capacity - p

            sl1 = slice(p, self.capacity)
            sl2 = slice(0, end % self.capacity)

            self.obs_buffer[sl1, :] = obs[:k]
            self.next_obs_buffer[sl1, :] = next_obs[:k]
            self.act_buffer[sl1, :] = act[:k]
            self.rew_buffer[sl1, :] = rew[:k]
            self.terminated_buffer[sl1, :] = terminated[:k]
            self.truncated_buffer[sl1, :] = truncated[:k]

            self.obs_buffer[sl2, :] = obs[k:]
            self.next_obs_buffer[sl2, :] = next_obs[k:]
            self.act_buffer[sl2, :] = act[k:]
            self.rew_buffer[sl2, :] = rew[k:]
            self.terminated_buffer[sl2, :] = terminated[k:]
            self.truncated_buffer[sl2, :] = truncated[k:]

        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + N)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> ReplayBatch:
        assert batch_size > 0, "batch_size must be > 0"
        assert self.size >= batch_size, f"Not enough samples: size={self.size}, batch_size={batch_size}"

        idx = np.random.randint(0, self.size, size=batch_size)

        obs = torch.from_numpy(self.obs_buffer[idx]).to(self.device, dtype=torch.float32)
        next_obs = torch.from_numpy(self.next_obs_buffer[idx]).to(self.device, dtype=torch.float32)
        act = torch.from_numpy(self.act_buffer[idx]).to(self.device, dtype=torch.float32)
        rew = torch.from_numpy(self.rew_buffer[idx]).to(self.device, dtype=torch.float32)

        terminated = torch.from_numpy(self.terminated_buffer[idx]).to(self.device, dtype=torch.bool)
        truncated = torch.from_numpy(self.truncated_buffer[idx]).to(self.device, dtype=torch.bool)

        return ReplayBatch(
            obs,
            next_obs,
            act,
            rew,
            terminated,
            truncated
        )
