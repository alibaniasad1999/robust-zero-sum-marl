"""Tests for src.networks.mlp — MLP building blocks."""

import torch
import torch.nn as nn
import pytest

from src.networks.mlp import mlp, MLPActor, MLPQFunction, AdversarialMLPQFunction, count_vars


class TestMLP:
    def test_output_shape(self):
        net = mlp([4, 32, 16, 2], nn.ReLU)
        x = torch.randn(8, 4)
        assert net(x).shape == (8, 2)

    def test_single_hidden(self):
        net = mlp([3, 5, 1], nn.Tanh)
        x = torch.randn(1, 3)
        assert net(x).shape == (1, 1)

    def test_output_activation(self):
        net = mlp([4, 8, 1], nn.ReLU, output_activation=nn.Sigmoid)
        x = torch.randn(10, 4)
        out = net(x)
        assert (out >= 0).all() and (out <= 1).all()


class TestMLPActor:
    @pytest.fixture
    def actor(self):
        return MLPActor(obs_dim=8, act_dim=3, hidden_sizes=(64, 64),
                        activation=nn.Tanh, act_limit=2.0)

    def test_output_shape(self, actor):
        obs = torch.randn(4, 8)
        out = actor(obs)
        assert out.shape == (4, 3)

    def test_output_bounded(self, actor):
        obs = torch.randn(100, 8)
        out = actor(obs)
        assert out.abs().max().item() <= 2.0 + 1e-6

    def test_single_obs(self, actor):
        obs = torch.randn(8)
        out = actor(obs)
        assert out.shape == (3,)

    def test_deterministic(self, actor):
        obs = torch.randn(1, 8)
        a1 = actor(obs)
        a2 = actor(obs)
        assert torch.allclose(a1, a2)


class TestMLPQFunction:
    @pytest.fixture
    def q(self):
        return MLPQFunction(obs_dim=8, act_dim=3, hidden_sizes=(64, 64),
                            activation=nn.ReLU)

    def test_output_shape(self, q):
        obs = torch.randn(4, 8)
        act = torch.randn(4, 3)
        out = q(obs, act)
        assert out.shape == (4,)

    def test_scalar_output(self, q):
        obs = torch.randn(1, 8)
        act = torch.randn(1, 3)
        out = q(obs, act)
        assert out.shape == (1,)

    def test_gradient_flows(self, q):
        obs = torch.randn(4, 8, requires_grad=True)
        act = torch.randn(4, 3, requires_grad=True)
        loss = q(obs, act).mean()
        loss.backward()
        assert obs.grad is not None
        assert act.grad is not None


class TestAdversarialMLPQFunction:
    @pytest.fixture
    def q(self):
        return AdversarialMLPQFunction(obs_dim=8, act_dim=3, dist_dim=3,
                                       hidden_sizes=(64, 64), activation=nn.ReLU)

    def test_output_shape(self, q):
        obs = torch.randn(4, 8)
        act = torch.randn(4, 3)
        act_d = torch.randn(4, 3)
        out = q(obs, act, act_d)
        assert out.shape == (4,)

    def test_different_dist_dim(self):
        q = AdversarialMLPQFunction(obs_dim=8, act_dim=3, dist_dim=5,
                                    hidden_sizes=(32,), activation=nn.ReLU)
        out = q(torch.randn(2, 8), torch.randn(2, 3), torch.randn(2, 5))
        assert out.shape == (2,)

    def test_gradient_flows_to_all_inputs(self, q):
        obs = torch.randn(4, 8, requires_grad=True)
        act = torch.randn(4, 3, requires_grad=True)
        act_d = torch.randn(4, 3, requires_grad=True)
        loss = q(obs, act, act_d).mean()
        loss.backward()
        assert obs.grad is not None
        assert act.grad is not None
        assert act_d.grad is not None


class TestCountVars:
    def test_linear(self):
        layer = nn.Linear(4, 3)  # 4*3 + 3 = 15
        assert count_vars(layer) == 15

    def test_actor(self):
        actor = MLPActor(4, 2, (16,), nn.Tanh, 1.0)
        assert count_vars(actor) > 0
