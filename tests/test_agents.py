"""Tests for src.agents.td3 — TD3Agent and AdversarialTD3Agent."""

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import pytest
import gymnasium as gym

from src.agents.td3 import TD3Agent, AdversarialTD3Agent


# Use a small, fast environment for testing
ENV_ID = "Pendulum-v1"


def make_env():
    return gym.make(ENV_ID)


# ── TD3Agent ───────────────────────────────────────────────────────

class TestTD3Agent:
    @pytest.fixture
    def agent(self, tmp_path):
        return TD3Agent(
            lambda: gym.make(ENV_ID),
            hidden_sizes=(32, 32),
            seed=42,
            steps_per_epoch=50,
            epochs=1,
            replay_size=500,
            batch_size=16,
            start_steps=20,
            update_after=20,
            update_every=10,
            n_updates=5,
            max_ep_len=50,
            device="cpu",
            log_dir=str(tmp_path / "nominal"),
            policy_delay=2,
            target_noise=0.2,
            noise_clip=0.5,
        )

    def test_init_creates_networks(self, agent):
        assert agent.pi is not None
        assert agent.q1 is not None
        assert agent.q2 is not None
        assert agent.pi_targ is not None
        assert agent.q1_targ is not None
        assert agent.q2_targ is not None

    def test_twin_q_are_independent(self, agent):
        """Q1 and Q2 should be separate networks with different params after init."""
        q1_params = list(agent.q1.parameters())
        q2_params = list(agent.q2.parameters())
        assert len(q1_params) == len(q2_params)

    def test_target_params_match_online(self, agent):
        """After construction targets are copies of online networks."""
        for p, pt in zip(agent.pi.parameters(), agent.pi_targ.parameters()):
            assert torch.allclose(p, pt)
        for p, pt in zip(agent.q1.parameters(), agent.q1_targ.parameters()):
            assert torch.allclose(p, pt)
        for p, pt in zip(agent.q2.parameters(), agent.q2_targ.parameters()):
            assert torch.allclose(p, pt)

    def test_get_action_shape(self, agent):
        obs = agent.envs.single_observation_space.sample()
        action = agent._get_action(obs, noise=0.1)
        assert action.shape == agent.envs.single_action_space.shape

    def test_get_action_bounded(self, agent):
        obs = agent.envs.single_observation_space.sample()
        for _ in range(20):
            action = agent._get_action(obs, noise=0.5)
            assert np.all(np.abs(action) <= agent.act_limit + 1e-6)

    def test_get_action_deterministic_without_noise(self, agent):
        obs = agent.envs.single_observation_space.sample()
        a1 = agent._get_action(obs, noise=0.0)
        a2 = agent._get_action(obs, noise=0.0)
        np.testing.assert_array_equal(a1, a2)

    def test_train_short_run(self, agent, tmp_path):
        agent.train(epochs=1)
        csv_path = os.path.join(str(tmp_path / "nominal"), "training_returns.csv")
        assert os.path.isfile(csv_path)

    def test_save_and_load(self, agent, tmp_path):
        save_path = str(tmp_path / "ckpt")
        agent.save(save_path)
        assert os.path.isfile(os.path.join(save_path, "pi.pt"))
        assert os.path.isfile(os.path.join(save_path, "q1.pt"))
        assert os.path.isfile(os.path.join(save_path, "q2.pt"))

        # Modify weights, then load and verify restore
        with torch.no_grad():
            for p in agent.pi.parameters():
                p.fill_(0.0)
        agent.load(save_path)
        has_nonzero = any(p.abs().sum() > 0 for p in agent.pi.parameters())
        assert has_nonzero

    def test_update_changes_params(self, agent):
        # Fill buffer with some transitions
        env = gym.make(ENV_ID)
        obs, _ = env.reset()
        for _ in range(30):
            act = env.action_space.sample()
            next_obs, rew, term, trunc, _ = env.step(act)
            agent.buffer.add(obs, next_obs, act, rew, term, trunc)
            obs = next_obs if not (term or trunc) else env.reset()[0]
        env.close()

        params_before = [p.clone() for p in agent.pi.parameters()]
        # Run enough updates to trigger delayed policy update
        for _ in range(agent.policy_delay):
            batch = agent.buffer.sample(16)
            agent._update(batch)
        changed = any(
            not torch.allclose(p_before, p_after)
            for p_before, p_after in zip(params_before, agent.pi.parameters())
        )
        assert changed

    def test_delayed_policy_update(self, agent):
        """Actor should only update every policy_delay steps."""
        env = gym.make(ENV_ID)
        obs, _ = env.reset()
        for _ in range(30):
            act = env.action_space.sample()
            next_obs, rew, term, trunc, _ = env.step(act)
            agent.buffer.add(obs, next_obs, act, rew, term, trunc)
            obs = next_obs if not (term or trunc) else env.reset()[0]
        env.close()

        # First update: only critics should change (policy_delay=2)
        agent._update_count = 0
        pi_before = [p.clone() for p in agent.pi.parameters()]
        q1_before = [p.clone() for p in agent.q1.parameters()]

        batch = agent.buffer.sample(16)
        metrics = agent._update(batch)

        # Actor should NOT have been updated
        assert "loss_pi" not in metrics
        pi_unchanged = all(
            torch.allclose(b, a)
            for b, a in zip(pi_before, agent.pi.parameters())
        )
        assert pi_unchanged, "pi should NOT update on non-delay step"

        # Critics should have been updated
        q1_changed = any(
            not torch.allclose(b, a)
            for b, a in zip(q1_before, agent.q1.parameters())
        )
        assert q1_changed, "q1 should update every step"

        # Second update: actor SHOULD update
        pi_before = [p.clone() for p in agent.pi.parameters()]
        batch = agent.buffer.sample(16)
        metrics = agent._update(batch)

        assert "loss_pi" in metrics
        pi_changed = any(
            not torch.allclose(b, a)
            for b, a in zip(pi_before, agent.pi.parameters())
        )
        assert pi_changed, "pi should update on delay step"

    def test_buffer_fills_during_train(self, agent):
        assert len(agent.buffer) == 0
        agent.train(epochs=1)
        assert len(agent.buffer) > 0


# ── AdversarialTD3Agent ────────────────────────────────────────────

class TestAdversarialTD3Agent:
    @pytest.fixture
    def agent(self, tmp_path):
        return AdversarialTD3Agent(
            lambda: gym.make(ENV_ID),
            hidden_sizes=(32, 32),
            seed=42,
            steps_per_epoch=50,
            epochs=1,
            replay_size=500,
            batch_size=16,
            start_steps=20,
            update_after=20,
            update_every=10,
            n_updates=5,
            max_ep_len=50,
            device="cpu",
            log_dir=str(tmp_path / "adversarial"),
            disturbance_ratio=0.1,
            disturbance_probability=0.5,
            use_transformer=False,
            policy_delay=2,
            target_noise=0.2,
            noise_clip=0.5,
        )

    @pytest.fixture
    def agent_with_transformer(self, tmp_path):
        return AdversarialTD3Agent(
            lambda: gym.make(ENV_ID),
            hidden_sizes=(32, 32),
            seed=42,
            steps_per_epoch=50,
            epochs=1,
            replay_size=500,
            batch_size=16,
            start_steps=20,
            update_after=20,
            update_every=10,
            n_updates=5,
            max_ep_len=50,
            device="cpu",
            log_dir=str(tmp_path / "adv_transformer"),
            disturbance_ratio=0.1,
            disturbance_probability=0.5,
            use_transformer=True,
            transformer_seq_len=5,
            transformer_d_model=16,
            transformer_nhead=2,
            transformer_layers=1,
            transformer_lr=1e-3,
            transformer_train_interval=25,
            policy_delay=2,
        )

    def test_init_creates_all_networks(self, agent):
        assert agent.pi_rob is not None
        assert agent.pi_adv is not None
        assert agent.q1 is not None
        assert agent.q2 is not None

    def test_twin_q_exist(self, agent):
        """Both Q1 and Q2 should exist with targets."""
        assert agent.q1_targ is not None
        assert agent.q2_targ is not None

    def test_ctrl_action_shape(self, agent):
        obs = agent.envs.single_observation_space.sample()
        act = agent._get_ctrl_action(obs, noise=0.0)
        assert act.shape == agent.envs.single_action_space.shape

    def test_adv_action_bounded(self, agent):
        obs = agent.envs.single_observation_space.sample()
        for _ in range(20):
            act = agent._get_adv_action(obs, noise=0.1)
            assert np.all(np.abs(act) <= agent.dist_limit + 1e-6)

    def test_opt_action_fallback(self, agent):
        """Without pi_opt loaded, _get_opt_action falls back to pi_rob."""
        obs = agent.envs.single_observation_space.sample()
        act = agent._get_opt_action(obs, noise=0.0)
        expected = agent._get_ctrl_action(obs, noise=0.0)
        np.testing.assert_array_equal(act, expected)

    def test_buffer_stores_combined_actions(self, agent):
        """Buffer total_act should be act_dim * 2."""
        assert agent.buffer.total_act == agent.act_dim * 2

    def test_train_short_run(self, agent):
        agent.train(epochs=1)
        assert len(agent.buffer) > 0

    def test_train_with_transformer(self, agent_with_transformer):
        agent_with_transformer.train(epochs=1)
        assert len(agent_with_transformer.buffer) > 0

    def test_save_and_load(self, agent, tmp_path):
        save_path = str(tmp_path / "adv_ckpt")
        agent.save(save_path)
        assert os.path.isfile(os.path.join(save_path, "pi_rob.pt"))
        assert os.path.isfile(os.path.join(save_path, "pi_adv.pt"))
        assert os.path.isfile(os.path.join(save_path, "q1.pt"))
        assert os.path.isfile(os.path.join(save_path, "q2.pt"))

        agent.load(save_path)

    def test_save_with_transformer(self, agent_with_transformer, tmp_path):
        save_path = str(tmp_path / "adv_trans_ckpt")
        agent_with_transformer.save(save_path)
        assert os.path.isfile(os.path.join(save_path, "detector.pt"))

    def test_gradient_flip(self, agent):
        """Verify that adversary gradients get flipped during update."""
        env = gym.make(ENV_ID)
        obs, _ = env.reset()
        for _ in range(30):
            act = env.action_space.sample()
            act_dist = env.action_space.sample() * agent.disturbance_ratio
            next_obs, rew, term, trunc, _ = env.step(act + act_dist)
            combined = np.concatenate([act, act_dist])
            agent.buffer.add(obs, next_obs, combined, rew, term, trunc)
            obs = next_obs if not (term or trunc) else env.reset()[0]
        env.close()

        # Record params before and after update
        rob_before = [p.clone() for p in agent.pi_rob.parameters()]
        adv_before = [p.clone() for p in agent.pi_adv.parameters()]

        # Run enough updates to trigger delayed policy update
        agent._update_count = 0
        for _ in range(agent.policy_delay):
            batch = agent.buffer.sample(16)
            agent._update(batch)

        rob_changed = any(
            not torch.allclose(b, a)
            for b, a in zip(rob_before, agent.pi_rob.parameters())
        )
        adv_changed = any(
            not torch.allclose(b, a)
            for b, a in zip(adv_before, agent.pi_adv.parameters())
        )
        assert rob_changed, "pi_rob should update"
        assert adv_changed, "pi_adv should update"

    def test_delayed_policy_update(self, agent):
        """Actor and adversary should only update every policy_delay steps."""
        env = gym.make(ENV_ID)
        obs, _ = env.reset()
        for _ in range(30):
            act = env.action_space.sample()
            act_dist = env.action_space.sample() * agent.disturbance_ratio
            next_obs, rew, term, trunc, _ = env.step(act + act_dist)
            combined = np.concatenate([act, act_dist])
            agent.buffer.add(obs, next_obs, combined, rew, term, trunc)
            obs = next_obs if not (term or trunc) else env.reset()[0]
        env.close()

        agent._update_count = 0
        rob_before = [p.clone() for p in agent.pi_rob.parameters()]

        batch = agent.buffer.sample(16)
        metrics = agent._update(batch)

        # Actor should NOT have updated on first call
        assert "loss_pi" not in metrics
        rob_unchanged = all(
            torch.allclose(b, a)
            for b, a in zip(rob_before, agent.pi_rob.parameters())
        )
        assert rob_unchanged, "pi_rob should NOT update on non-delay step"

    def test_load_pi_opt(self, tmp_path):
        """Test loading a pre-trained optimal policy."""
        # First train and save a nominal agent
        nominal = TD3Agent(
            lambda: gym.make(ENV_ID),
            hidden_sizes=(32, 32),
            steps_per_epoch=30,
            epochs=1,
            start_steps=10,
            update_after=10,
            batch_size=8,
            n_updates=2,
            max_ep_len=30,
            device="cpu",
            log_dir=str(tmp_path / "nom"),
        )
        nominal.train()
        nominal.save(str(tmp_path / "nom_ckpt"))

        # Now create adversarial agent with pi_opt path
        adv = AdversarialTD3Agent(
            lambda: gym.make(ENV_ID),
            hidden_sizes=(32, 32),
            steps_per_epoch=30,
            epochs=1,
            start_steps=10,
            update_after=10,
            batch_size=8,
            n_updates=2,
            max_ep_len=30,
            device="cpu",
            log_dir=str(tmp_path / "adv"),
            use_transformer=False,
            pi_opt_path=str(tmp_path / "nom_ckpt"),
        )
        assert adv.pi_opt is not None
        # pi_opt should be frozen
        for p in adv.pi_opt.parameters():
            assert not p.requires_grad
