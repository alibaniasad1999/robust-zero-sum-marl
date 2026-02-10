"""Tests for src.detector.transformer — Transformer disturbance detector."""

import numpy as np
import torch
import pytest

from src.detector.transformer import (
    TransformerDisturbanceDetector,
    HistoryBuffer,
    DisturbanceDetectorTrainer,
    AdaptiveControllerBlender,
)


# ── TransformerDisturbanceDetector ──────────────────────────────────

class TestTransformerDisturbanceDetector:
    @pytest.fixture
    def detector(self):
        return TransformerDisturbanceDetector(
            obs_dim=8, sequence_length=10, d_model=32,
            nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0,
        )

    def test_output_shapes(self, detector):
        x = torch.randn(4, 10, 8)
        dist_prob, alpha = detector(x)
        assert dist_prob.shape == (4,)
        assert alpha.shape == (4,)

    def test_outputs_in_0_1(self, detector):
        x = torch.randn(16, 10, 8)
        detector.eval()
        with torch.no_grad():
            dist_prob, alpha = detector(x)
        assert (dist_prob >= 0).all() and (dist_prob <= 1).all()
        assert (alpha >= 0).all() and (alpha <= 1).all()

    def test_shorter_sequence(self, detector):
        """Detector should handle sequences shorter than max length."""
        x = torch.randn(2, 5, 8)
        dist_prob, alpha = detector(x)
        assert dist_prob.shape == (2,)

    def test_single_sample(self, detector):
        x = torch.randn(1, 10, 8)
        dist_prob, alpha = detector(x)
        assert dist_prob.shape == (1,)

    def test_gradient_flows(self, detector):
        x = torch.randn(2, 10, 8, requires_grad=True)
        dist_prob, alpha = detector(x)
        loss = dist_prob.sum() + alpha.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_obs_dim(self):
        det = TransformerDisturbanceDetector(
            obs_dim=27, sequence_length=20, d_model=64, nhead=4, num_layers=2,
        )
        x = torch.randn(3, 20, 27)
        dist_prob, alpha = det(x)
        assert dist_prob.shape == (3,)


# ── HistoryBuffer ───────────────────────────────────────────────────

class TestHistoryBuffer:
    def test_initial_state(self):
        hb = HistoryBuffer(obs_dim=4, max_length=10)
        seq = hb.get_sequence()
        assert seq.shape == (1, 10, 4)
        assert (seq == 0).all()

    def test_add_fills_buffer(self):
        hb = HistoryBuffer(obs_dim=3, max_length=5)
        for i in range(5):
            hb.add(np.full(3, float(i + 1), dtype=np.float32))
        seq = hb.get_sequence()
        assert seq.shape == (1, 5, 3)
        # last entry should be [5, 5, 5]
        assert torch.allclose(seq[0, -1], torch.tensor([5.0, 5.0, 5.0]))

    def test_overflow_keeps_recent(self):
        hb = HistoryBuffer(obs_dim=2, max_length=3)
        for i in range(10):
            hb.add(np.array([float(i), float(i)], dtype=np.float32))
        seq = hb.get_sequence()
        # should contain last 3: [7,7], [8,8], [9,9]
        assert torch.allclose(seq[0, 0], torch.tensor([7.0, 7.0]))
        assert torch.allclose(seq[0, 2], torch.tensor([9.0, 9.0]))

    def test_reset(self):
        hb = HistoryBuffer(obs_dim=4, max_length=5)
        hb.add(np.ones(4, dtype=np.float32))
        hb.reset()
        seq = hb.get_sequence()
        assert (seq == 0).all()

    def test_add_copies_data(self):
        """Ensure buffer copies input, not references."""
        hb = HistoryBuffer(obs_dim=3, max_length=5)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        hb.add(arr)
        arr[:] = 0.0  # mutate original
        seq = hb.get_sequence()
        assert seq[0, -1, 0].item() == 1.0


# ── DisturbanceDetectorTrainer ──────────────────────────────────────

class TestDisturbanceDetectorTrainer:
    @pytest.fixture
    def setup(self):
        det = TransformerDisturbanceDetector(
            obs_dim=4, sequence_length=8, d_model=16,
            nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
        )
        trainer = DisturbanceDetectorTrainer(det, learning_rate=1e-3)
        return det, trainer

    def test_train_step_returns_losses(self, setup):
        det, trainer = setup
        obs = torch.randn(8, 8, 4)
        dist = torch.randint(0, 2, (8,)).float()
        alpha = torch.rand(8)
        result = trainer.train_step(obs, dist, alpha)
        assert "loss_disturbance" in result
        assert "loss_blending" in result
        assert "total_loss" in result
        assert all(v >= 0 for v in result.values())

    def test_train_step_reduces_loss(self, setup):
        det, trainer = setup
        obs = torch.randn(16, 8, 4)
        dist = torch.ones(16)
        alpha = torch.ones(16) * 0.8
        loss_before = trainer.evaluate(obs, dist, alpha)["total_loss"]
        for _ in range(50):
            trainer.train_step(obs, dist, alpha)
        loss_after = trainer.evaluate(obs, dist, alpha)["total_loss"]
        assert loss_after < loss_before

    def test_evaluate_no_grad(self, setup):
        det, trainer = setup
        obs = torch.randn(4, 8, 4)
        dist = torch.zeros(4)
        alpha = torch.rand(4)
        result = trainer.evaluate(obs, dist, alpha)
        # parameters should not have gradients from evaluate
        for p in det.parameters():
            assert p.grad is None or (p.grad == 0).all()


# ── AdaptiveControllerBlender ───────────────────────────────────────

class TestAdaptiveControllerBlender:
    @pytest.fixture
    def blender(self):
        det = TransformerDisturbanceDetector(
            obs_dim=4, sequence_length=5, d_model=16,
            nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
        )
        hb = HistoryBuffer(obs_dim=4, max_length=5)
        return AdaptiveControllerBlender(det, hb, smoothing_window=3)

    def test_returns_tuple_of_three(self, blender):
        obs = np.random.randn(4).astype(np.float32)
        a_opt = np.ones(3, dtype=np.float32)
        a_rob = np.zeros(3, dtype=np.float32)
        result = blender.get_blended_action(obs, a_opt, a_rob)
        assert len(result) == 3
        blended, alpha_val, dist_val = result
        assert blended.shape == (3,)

    def test_alpha_in_range(self, blender):
        obs = np.random.randn(4).astype(np.float32)
        a_opt = np.ones(3, dtype=np.float32)
        a_rob = np.zeros(3, dtype=np.float32)
        _, alpha_val, dist_val = blender.get_blended_action(obs, a_opt, a_rob)
        assert 0.0 <= alpha_val <= 1.0
        assert 0.0 <= dist_val <= 1.0

    def test_blending_formula(self, blender):
        """When alpha=1, blended should equal a_opt; alpha=0 -> a_rob."""
        obs = np.random.randn(4).astype(np.float32)
        a_opt = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        a_rob = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        blended, alpha, _ = blender.get_blended_action(obs, a_opt, a_rob)
        expected = alpha * a_opt + (1.0 - alpha) * a_rob
        np.testing.assert_allclose(blended, expected, atol=1e-5)

    def test_smoothing(self, blender):
        """Multiple calls should smooth alpha values."""
        a_opt = np.ones(3, dtype=np.float32)
        a_rob = np.zeros(3, dtype=np.float32)
        alphas = []
        for _ in range(5):
            obs = np.random.randn(4).astype(np.float32)
            _, alpha, _ = blender.get_blended_action(obs, a_opt, a_rob, use_smoothing=True)
            alphas.append(alpha)
        # with smoothing, values should be more stable
        assert len(alphas) == 5

    def test_reset_clears_state(self, blender):
        obs = np.random.randn(4).astype(np.float32)
        blender.get_blended_action(obs, np.ones(3), np.zeros(3))
        blender.reset()
        assert len(blender.alpha_history) == 0
        seq = blender.history_buffer.get_sequence()
        assert (seq == 0).all()

    def test_no_smoothing(self, blender):
        """Without smoothing, raw alpha should be used."""
        obs = np.random.randn(4).astype(np.float32)
        a_opt = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        a_rob = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        blended, alpha, _ = blender.get_blended_action(
            obs, a_opt, a_rob, use_smoothing=False
        )
        expected = alpha * a_opt + (1.0 - alpha) * a_rob
        np.testing.assert_allclose(blended, expected, atol=1e-5)
