"""Tests for src.utils.logger — DatasetLogger and TransitionRecord."""

import os
import csv
import tempfile

import numpy as np
import pytest

from src.utils.logger import DatasetLogger, TransitionRecord


def _make_record(t=0, episode_id=0, obs_dim=4, act_dim=2, tag="none"):
    return TransitionRecord(
        t=t,
        episode_id=episode_id,
        obs=np.random.randn(obs_dim).astype(np.float32),
        action_ctrl=np.random.randn(act_dim).astype(np.float32),
        action_dist=np.random.randn(act_dim).astype(np.float32),
        action_total=np.random.randn(act_dim).astype(np.float32),
        reward=float(np.random.randn()),
        terminated=False,
        truncated=False,
        disturbance_params=np.array([0.0, 0.1], dtype=np.float32),
        disturbance_tag=tag,
        robust_weight_target=0.5,
    )


class TestTransitionRecord:
    def test_default_values(self):
        r = _make_record()
        assert r.disturbance_tag == "none"
        assert r.robust_weight_target == 0.0 or r.robust_weight_target == 0.5

    def test_fields_accessible(self):
        r = _make_record(t=5, episode_id=3, tag="adversary")
        assert r.t == 5
        assert r.episode_id == 3
        assert r.disturbance_tag == "adversary"


class TestDatasetLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return DatasetLogger(str(tmp_path / "dataset"))

    def test_creates_log_dir(self, logger):
        assert os.path.isdir(logger.log_dir)

    def test_end_episode_empty_buffer_noop(self, logger):
        logger.end_episode()  # should not crash
        episode_dir = os.path.join(logger.log_dir, "episodes")
        assert not os.path.exists(episode_dir)

    def test_single_episode_saves_npz(self, logger):
        for t in range(10):
            logger.log(_make_record(t=t, episode_id=0))
        logger.end_episode()

        episode_dir = os.path.join(logger.log_dir, "episodes")
        npz_path = os.path.join(episode_dir, "ep_000000.npz")
        assert os.path.isfile(npz_path)

        data = np.load(npz_path, allow_pickle=True)
        assert data["obs"].shape == (10, 4)
        assert data["action_ctrl"].shape == (10, 2)
        assert data["reward"].shape == (10,)
        assert data["terminated"].shape == (10,)

    def test_csv_summary_written(self, logger):
        for t in range(5):
            logger.log(_make_record(t=t))
        logger.end_episode()

        csv_path = os.path.join(logger.log_dir, "transitions.csv")
        assert os.path.isfile(csv_path)

        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["episode_id", "length", "total_reward",
                           "has_disturbance", "mean_alpha_target"]
        assert len(rows) == 2  # header + 1 episode

    def test_multiple_episodes_increment_id(self, logger):
        for ep in range(3):
            for t in range(5):
                logger.log(_make_record(t=t, episode_id=ep))
            logger.end_episode()

        episode_dir = os.path.join(logger.log_dir, "episodes")
        assert os.path.isfile(os.path.join(episode_dir, "ep_000000.npz"))
        assert os.path.isfile(os.path.join(episode_dir, "ep_000001.npz"))
        assert os.path.isfile(os.path.join(episode_dir, "ep_000002.npz"))

    def test_csv_has_correct_disturbance_flag(self, logger):
        # Episode with disturbance
        for t in range(3):
            logger.log(_make_record(t=t, tag="adversary"))
        logger.end_episode()

        # Episode without
        for t in range(3):
            logger.log(_make_record(t=t, tag="none"))
        logger.end_episode()

        csv_path = os.path.join(logger.log_dir, "transitions.csv")
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert rows[1][3] == "1"  # has_disturbance = True
        assert rows[2][3] == "0"  # has_disturbance = False

    def test_buffer_cleared_after_end_episode(self, logger):
        logger.log(_make_record())
        logger.end_episode()
        # buffer should be empty now — another end_episode is a no-op
        logger.end_episode()
        episode_dir = os.path.join(logger.log_dir, "episodes")
        files = os.listdir(episode_dir)
        assert len(files) == 1  # only the first episode


class TestExtractWindows:
    def test_basic_extraction(self, tmp_path):
        logger = DatasetLogger(str(tmp_path / "dataset"))
        obs_dim, act_dim, seq_len = 4, 2, 5

        # Log 2 episodes of 10 steps each
        for ep in range(2):
            for t in range(10):
                logger.log(_make_record(t=t, episode_id=ep, obs_dim=obs_dim, act_dim=act_dim))
            logger.end_episode()

        episode_dir = os.path.join(logger.log_dir, "episodes")
        obs_w, dist_l, alpha_l = DatasetLogger.extract_windows(episode_dir, seq_len)

        # 2 episodes * 10 steps = 20 windows
        assert obs_w.shape == (20, seq_len, obs_dim)
        assert dist_l.shape == (20,)
        assert alpha_l.shape == (20,)

    def test_windows_are_zero_padded(self, tmp_path):
        logger = DatasetLogger(str(tmp_path / "dataset"))
        seq_len = 5

        for t in range(3):
            logger.log(_make_record(t=t, obs_dim=4, act_dim=2))
        logger.end_episode()

        episode_dir = os.path.join(logger.log_dir, "episodes")
        obs_w, _, _ = DatasetLogger.extract_windows(episode_dir, seq_len)

        # First window: obs at t=0, padded with 4 zeros before it
        first_window = obs_w[0]
        assert first_window.shape == (seq_len, 4)
        # First 4 rows should be zero padding
        np.testing.assert_array_equal(first_window[:4], np.zeros((4, 4)))

    def test_empty_dir(self, tmp_path):
        empty_dir = str(tmp_path / "empty_episodes")
        os.makedirs(empty_dir)
        obs_w, dist_l, alpha_l = DatasetLogger.extract_windows(empty_dir, 5)
        assert obs_w.shape[0] == 0
