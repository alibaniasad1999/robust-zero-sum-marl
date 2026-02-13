"""MLP network building blocks for TD3 actor-critic architectures."""

from typing import List, Optional, Type

import numpy as np
import torch
import torch.nn as nn


def mlp(
    sizes: List[int],
    activation: Type[nn.Module],
    output_activation: Optional[Type[nn.Module]] = None,
) -> nn.Sequential:
    """Build a feedforward MLP from a list of layer sizes."""
    if output_activation is None:
        output_activation = nn.Identity
    layers: list[nn.Module] = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module: nn.Module) -> int:
    return sum(np.prod(p.shape) for p in module.parameters())


class MLPActor(nn.Module):
    """Deterministic policy: obs -> tanh-squashed action scaled by act_limit."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        activation: Type[nn.Module],
        act_limit: float,
    ):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):
    """Q-function for single-agent: Q(obs, act) -> scalar."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        activation: Type[nn.Module],
    ):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.q(torch.cat([obs, act], dim=-1)), -1)


class AdversarialMLPQFunction(nn.Module):
    """Q-function for two-player game: Q(obs, act_ctrl, act_dist) -> scalar."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        dist_dim: int,
        hidden_sizes: tuple[int, ...],
        activation: Type[nn.Module],
    ):
        super().__init__()
        self.q = mlp(
            [obs_dim + act_dim + dist_dim] + list(hidden_sizes) + [1], activation
        )

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor, act_d: torch.Tensor
    ) -> torch.Tensor:
        return torch.squeeze(self.q(torch.cat([obs, act, act_d], dim=-1)), -1)
