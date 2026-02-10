"""Tests for src.utils.buffers.replay_buffer."""

import numpy as np
import pytest
import torch

from src.utils.buffers.replay_buffer import ReplayBuffer, ReplayBatch


@pytest.fixture
def buf():
    return ReplayBuffer(
        capacity=100, obs_dim=4, total_act=2, device=torch.device("cpu"), dtype=np.float32
    )


class TestReplayBuffer:
    def test_add_single(self, buf: ReplayBuffer):
        obs = np.ones(4, dtype=np.float32)
        next_obs = np.ones(4, dtype=np.float32) * 2
        act = np.array([0.5, -0.5], dtype=np.float32)
        buf.add(obs, next_obs, act, rew=1.0, terminated=False, truncated=False)
        assert len(buf) == 1

    def test_sample(self, buf: ReplayBuffer):
        for i in range(10):
            buf.add(
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.random.randn(2).astype(np.float32),
                rew=float(i),
                terminated=False,
                truncated=False,
            )
        batch = buf.sample(5)
        assert isinstance(batch, ReplayBatch)
        assert batch.obs.shape == (5, 4)
        assert batch.act.shape == (5, 2)
        assert batch.rew.shape == (5, 1)

    def test_capacity_wrapping(self):
        small = ReplayBuffer(
            capacity=5, obs_dim=2, total_act=1, device=torch.device("cpu"), dtype=np.float32
        )
        for i in range(10):
            small.add(
                np.array([i, i], dtype=np.float32),
                np.array([i + 1, i + 1], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                rew=0.0,
                terminated=False,
                truncated=False,
            )
        assert len(small) == 5
        batch = small.sample(5)
        assert batch.obs.shape == (5, 2)

    def test_add_batch(self, buf: ReplayBuffer):
        N = 8
        buf.add_batch(
            obs=np.random.randn(N, 4).astype(np.float32),
            next_obs=np.random.randn(N, 4).astype(np.float32),
            act=np.random.randn(N, 2).astype(np.float32),
            rew=np.random.randn(N, 1).astype(np.float32),
            terminated=np.zeros((N, 1), dtype=np.bool_),
            truncated=np.zeros((N, 1), dtype=np.bool_),
        )
        assert len(buf) == N

    def test_done_property(self, buf: ReplayBuffer):
        buf.add(np.zeros(4), np.zeros(4), np.zeros(2), 0.0, True, False)
        buf.add(np.zeros(4), np.zeros(4), np.zeros(2), 0.0, False, True)
        buf.add(np.zeros(4), np.zeros(4), np.zeros(2), 0.0, False, False)
        # Check raw buffers directly (sample is random with replacement)
        term = buf.terminated_buffer[:3].flatten()
        trunc = buf.truncated_buffer[:3].flatten()
        done = term | trunc
        assert done.sum() == 2

    def test_sample_insufficient_raises(self):
        small = ReplayBuffer(
            capacity=5, obs_dim=2, total_act=1, device=torch.device("cpu"), dtype=np.float32
        )
        with pytest.raises(AssertionError):
            small.sample(1)
