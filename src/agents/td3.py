"""
TD3 agents for the zero-sum robust RL framework.

    TD3Agent                — trains pi_opt on the nominal environment
    AdversarialTD3Agent     — trains pi_rob + pi_adv in a zero-sum Markov game,
                              with optional transformer-based disturbance detection
                              and policy mixing

TD3 (Twin Delayed DDPG) extends DDPG with three improvements:
    1. Twin Q-networks (clipped double Q-learning) to reduce overestimation
    2. Delayed policy updates (actor updated less frequently than critics)
    3. Target policy smoothing (clipped noise on target actions)

Both agents support vectorized environments via ``num_envs`` (default 1).
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.networks.mlp import (
    MLPActor,
    MLPQFunction,
    AdversarialMLPQFunction,
    count_vars,
)
from src.utils.buffers.replay_buffer import ReplayBuffer, ReplayBatch
from src.utils.logger import DatasetLogger, TransitionRecord
from src.utils.perf_logger import PerformanceLogger
from src.detector.transformer import (
    TransformerDisturbanceDetector,
    HistoryBuffer,
    DisturbanceDetectorTrainer,
    AdaptiveControllerBlender,
)


def _make_vec_env(env_fn, num_envs: int):
    """Create a vectorized env from an env factory function."""
    if num_envs > 1:
        return gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])
    return gym.vector.SyncVectorEnv([env_fn])


def _grad_norm(module: nn.Module) -> float:
    """Compute total L2 gradient norm for a module's parameters."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ======================================================================
# TD3Agent  — single-agent, trains pi_opt on nominal environment
# ======================================================================
class TD3Agent:
    def __init__(
        self,
        env_fn,
        *,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
        seed: int = 0,
        steps_per_epoch: int = 4_000,
        epochs: int = 100,
        replay_size: int = 1_000_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        pi_lr: float = 3e-4,
        q_lr: float = 3e-4,
        batch_size: int = 256,
        start_steps: int = 25_000,
        update_after: int = 25_000,
        update_every: int = 50,
        n_updates: int = 500,
        act_noise: float = 0.1,
        max_ep_len: int = 1_000,
        save_freq: int = 10,
        device: str = "auto",
        log_dir: str = "logs/nominal",
        num_envs: int = 1,
        # TD3-specific
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Device is: {self.device}")

        # Vectorized environment
        self.num_envs = num_envs
        self.envs = _make_vec_env(env_fn, num_envs)

        obs_dim = self.envs.single_observation_space.shape[0]
        act_dim = self.envs.single_action_space.shape[0]
        self.act_limit = float(self.envs.single_action_space.high[0])

        # Networks — twin Q-functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes,
                           activation, self.act_limit).to(self.device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes,
                               activation).to(self.device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes,
                               activation).to(self.device)

        # Target networks
        self.pi_targ = deepcopy(self.pi)
        self.q1_targ = deepcopy(self.q1)
        self.q2_targ = deepcopy(self.q2)
        for p in (list(self.pi_targ.parameters())
                  + list(self.q1_targ.parameters())
                  + list(self.q2_targ.parameters())):
            p.requires_grad = False

        # Optimizers
        self.pi_optim = Adam(self.pi.parameters(), lr=pi_lr)
        self.q1_optim = Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optim = Adam(self.q2.parameters(), lr=q_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=replay_size,
            obs_dim=obs_dim,
            total_act=act_dim,
            device=self.device,
            dtype=np.float32,
            env_num=num_envs if num_envs > 1 else None,
        )

        # Hyperparams
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.n_updates = n_updates
        self.act_noise = act_noise
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._csv_path = os.path.join(log_dir, "training_returns.csv")

        # TD3-specific hyperparams
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self._update_count = 0

        # Performance logger (GPU vs CPU speed tracking)
        self.perf = PerformanceLogger(self.device, log_dir)

        print(f"[TD3Agent] device={self.device}  num_envs={num_envs}  "
              f"pi_params={count_vars(self.pi)}  "
              f"q1_params={count_vars(self.q1)}  "
              f"q2_params={count_vars(self.q2)}  "
              f"policy_delay={policy_delay}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_action(self, obs: np.ndarray, noise: float) -> np.ndarray:
        """Get action for obs of shape (N, obs_dim) or (obs_dim,)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = self.pi(obs_t).cpu().numpy()
        a += noise * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    # ------------------------------------------------------------------
    def _update(self, batch: ReplayBatch) -> Dict[str, float]:
        o, a, r = batch.obs, batch.act, batch.rew.squeeze(-1)
        o2 = batch.next_obs
        d = batch.done.float().squeeze(-1)

        # ---- Critic update (both Q1 and Q2) ----
        with torch.no_grad():
            # Target policy smoothing
            a2 = self.pi_targ(o2)
            epsilon = torch.randn_like(a2) * self.target_noise
            epsilon = epsilon.clamp(-self.noise_clip, self.noise_clip)
            a2 = (a2 + epsilon).clamp(-self.act_limit, self.act_limit)

            # Clipped double Q-learning
            q1_targ_val = self.q1_targ(o2, a2)
            q2_targ_val = self.q2_targ(o2, a2)
            q_targ_val = torch.min(q1_targ_val, q2_targ_val)
            backup = r + self.gamma * (1.0 - d) * q_targ_val

        # Q1
        q1_val = self.q1(o, a)
        loss_q1 = ((q1_val - backup) ** 2).mean()

        self.q1_optim.zero_grad()
        loss_q1.backward()
        grad_q1 = _grad_norm(self.q1)
        self.q1_optim.step()

        # Q2
        q2_val = self.q2(o, a)
        loss_q2 = ((q2_val - backup) ** 2).mean()

        self.q2_optim.zero_grad()
        loss_q2.backward()
        grad_q2 = _grad_norm(self.q2)
        self.q2_optim.step()

        metrics = {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "q1_val": q1_val.mean().item(),
            "q2_val": q2_val.mean().item(),
            "grad_q1": grad_q1,
            "grad_q2": grad_q2,
        }

        # ---- Delayed actor update ----
        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            for p in self.q1.parameters():
                p.requires_grad = False
            loss_pi = -self.q1(o, self.pi(o)).mean()
            self.pi_optim.zero_grad()
            loss_pi.backward()
            grad_pi = _grad_norm(self.pi)
            self.pi_optim.step()
            for p in self.q1.parameters():
                p.requires_grad = True

            metrics["loss_pi"] = loss_pi.item()
            metrics["grad_pi"] = grad_pi

            # Polyak update targets (only when policy is updated)
            with torch.no_grad():
                for p, pt in zip(self.pi.parameters(), self.pi_targ.parameters()):
                    pt.data.mul_(self.polyak).add_(
                        (1.0 - self.polyak) * p.data)
                for p, pt in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    pt.data.mul_(self.polyak).add_(
                        (1.0 - self.polyak) * p.data)
                for p, pt in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    pt.data.mul_(self.polyak).add_(
                        (1.0 - self.polyak) * p.data)

        return metrics

    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None) -> None:
        epochs = epochs or self.epochs
        N = self.num_envs

        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "episode_return"])

        # Metrics CSV
        metrics_csv = os.path.join(self.log_dir, "training_metrics.csv")
        if not os.path.isfile(metrics_csv):
            with open(metrics_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "epoch", "step", "loss_q1", "loss_q2", "loss_pi",
                    "q1_val", "q2_val",
                    "grad_q1", "grad_q2", "grad_pi",
                    "mean_return", "mean_length",
                    "episodes", "buffer_size", "elapsed_s",
                ])

        # Each loop iteration collects N transitions
        total_transitions = self.steps_per_epoch * epochs
        total_steps = total_transitions // N

        obs, _ = self.envs.reset()  # (N, obs_dim)
        ep_rets = np.zeros(N)
        ep_lens = np.zeros(N, dtype=int)
        t0 = time.time()

        # Epoch-level metric accumulators
        epoch_metrics: Dict[str, list] = defaultdict(list)
        epoch_returns: list[float] = []
        epoch_lengths: list[int] = []
        epoch_env_steps = 0
        epoch_update_count = 0

        self.perf.start_epoch()

        for t in range(total_steps):
            global_step = t * N

            # Action selection
            if global_step < self.start_steps:
                acts = np.stack([self.envs.single_action_space.sample()
                                 for _ in range(N)])
            else:
                acts = self._get_action(obs, self.act_noise)

            self.perf.start("env_step")
            next_obs, rews, terms, truncs, infos = self.envs.step(acts)
            self.perf.stop("env_step")
            epoch_env_steps += N

            # Use final_observation for terminal transitions (auto-reset)
            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                for i in range(N):
                    if infos["_final_observation"][i]:
                        real_next_obs[i] = infos["final_observation"][i]

            self.buffer.add_batch(
                obs, real_next_obs, acts,
                rews.reshape(-1, 1),
                terms.reshape(-1, 1),
                truncs.reshape(-1, 1),
            )

            # Per-env episode tracking
            ep_rets += rews
            ep_lens += 1
            dones = terms | truncs
            for i in range(N):
                if dones[i]:
                    with open(self._csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([global_step, ep_rets[i]])
                    epoch_returns.append(ep_rets[i])
                    epoch_lengths.append(ep_lens[i])
                    ep_rets[i] = 0.0
                    ep_lens[i] = 0

            obs = next_obs  # auto-reset already applied by VectorEnv

            # Network updates
            if global_step >= self.update_after and t % self.update_every == 0:
                for _ in range(self.n_updates):
                    self.perf.start("buffer_sample")
                    batch = self.buffer.sample(self.batch_size)
                    self.perf.stop("buffer_sample")
                    self.perf.start("update")
                    m = self._update(batch)
                    self.perf.stop("update")
                    epoch_update_count += 1
                    for k, v in m.items():
                        epoch_metrics[k].append(v)

            # Epoch boundary
            if (global_step + N) % self.steps_per_epoch < N:
                epoch = (global_step + N) // self.steps_per_epoch
                if epoch > 0:
                    elapsed = time.time() - t0
                    mean_ret = np.mean(
                        epoch_returns) if epoch_returns else float("nan")
                    mean_len = np.mean(
                        epoch_lengths) if epoch_lengths else float("nan")

                    # Compute epoch-averaged metrics
                    avg = {k: np.mean(v)
                           for k, v in epoch_metrics.items() if v}

                    print(f"\n[TD3Agent] Epoch {epoch}/{epochs}  "
                          f"elapsed={elapsed:.0f}s  "
                          f"buf={len(self.buffer)}")
                    print(f"  return   {mean_ret:>8.1f}  "
                          f"(episodes={len(epoch_returns)}  "
                          f"mean_len={mean_len:.0f})")
                    if avg:
                        print(f"  loss_q1  {avg.get('loss_q1', 0):>10.4f}   "
                              f"loss_q2  {avg.get('loss_q2', 0):>10.4f}   "
                              f"loss_pi  {avg.get('loss_pi', 0):>10.4f}")
                        print(f"  Q1_val   {avg.get('q1_val', 0):>10.2f}   "
                              f"Q2_val   {avg.get('q2_val', 0):>10.2f}")
                        print(f"  grad_q1  {avg.get('grad_q1', 0):>10.4f}   "
                              f"grad_q2  {avg.get('grad_q2', 0):>10.4f}   "
                              f"grad_pi  {avg.get('grad_pi', 0):>10.4f}")

                    # Log to metrics CSV
                    with open(metrics_csv, "a", newline="") as f:
                        csv.writer(f).writerow([
                            epoch, global_step,
                            f"{avg.get('loss_q1', ''):.6f}" if avg else "",
                            f"{avg.get('loss_q2', ''):.6f}" if avg else "",
                            f"{avg.get('loss_pi', ''):.6f}" if avg else "",
                            f"{avg.get('q1_val', ''):.4f}" if avg else "",
                            f"{avg.get('q2_val', ''):.4f}" if avg else "",
                            f"{avg.get('grad_q1', ''):.4f}" if avg else "",
                            f"{avg.get('grad_q2', ''):.4f}" if avg else "",
                            f"{avg.get('grad_pi', ''):.4f}" if avg else "",
                            f"{mean_ret:.2f}" if not np.isnan(
                                mean_ret) else "",
                            f"{mean_len:.1f}" if not np.isnan(
                                mean_len) else "",
                            len(epoch_returns),
                            len(self.buffer),
                            f"{elapsed:.1f}",
                        ])

                    # Reset epoch accumulators
                    epoch_metrics = defaultdict(list)
                    epoch_returns = []
                    epoch_lengths = []

                    # Performance summary
                    self.perf.epoch_summary(
                        epoch, global_step,
                        steps_this_epoch=epoch_env_steps,
                        updates_this_epoch=epoch_update_count,
                    )
                    epoch_env_steps = 0
                    epoch_update_count = 0
                    self.perf.start_epoch()

                    if epoch % self.save_freq == 0:
                        self.save()

    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.log_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)
        torch.save(self.pi.state_dict(), os.path.join(path, "pi.pt"))
        torch.save(self.q1.state_dict(), os.path.join(path, "q1.pt"))
        torch.save(self.q2.state_dict(), os.path.join(path, "q2.pt"))
        print(f"  [saved] {path}")

    def load(self, path: str) -> None:
        self.pi.load_state_dict(torch.load(os.path.join(
            path, "pi.pt"), map_location=self.device))
        self.q1.load_state_dict(torch.load(os.path.join(
            path, "q1.pt"), map_location=self.device))
        self.q2.load_state_dict(torch.load(os.path.join(
            path, "q2.pt"), map_location=self.device))
        self.pi_targ = deepcopy(self.pi)
        self.q1_targ = deepcopy(self.q1)
        self.q2_targ = deepcopy(self.q2)
        print(f"  [loaded] {path}")


# ======================================================================
# AdversarialTD3Agent — zero-sum game: pi_rob vs pi_adv
# ======================================================================
class AdversarialTD3Agent:
    """
    Two-player zero-sum adversarial TD3.

    The *controller* (pi_rob) maximises cumulative reward.
    The *adversary* (pi_adv) minimises it (gradient flip).
    Optionally trains a transformer disturbance detector and uses it
    to blend pi_opt (from a separately-trained TD3Agent) with pi_rob.

    TD3 improvements over DDPG:
        - Twin Q-networks (clipped double Q-learning)
        - Delayed policy updates (every ``policy_delay`` critic updates)
        - Target policy smoothing (clipped noise on target actions)
    """

    def __init__(
        self,
        env_fn,
        *,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
        seed: int = 0,
        steps_per_epoch: int = 4_000,
        epochs: int = 100,
        replay_size: int = 1_000_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        pi_lr: float = 3e-4,
        q_lr: float = 3e-4,
        batch_size: int = 256,
        start_steps: int = 25_000,
        update_after: int = 25_000,
        update_every: int = 50,
        n_updates: int = 500,
        act_noise: float = 0.1,
        max_ep_len: int = 1_000,
        save_freq: int = 10,
        device: str = "auto",
        log_dir: str = "logs/adversarial",
        # adversary
        disturbance_ratio: float = 0.15,
        disturbance_probability: float = 0.5,
        # transformer detector
        use_transformer: bool = True,
        transformer_seq_len: int = 10,
        transformer_d_model: int = 64,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_lr: float = 3e-4,
        transformer_train_interval: int = 500,
        # pre-trained optimal policy (for blending)
        pi_opt_path: Optional[str] = None,
        num_envs: int = 1,
        # TD3-specific
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Vectorized environment
        self.num_envs = num_envs
        self.envs = _make_vec_env(env_fn, num_envs)

        obs_dim = self.envs.single_observation_space.shape[0]
        act_dim = self.envs.single_action_space.shape[0]
        self.act_limit = float(self.envs.single_action_space.high[0])
        self.dist_limit = self.act_limit * disturbance_ratio

        # ---- networks: twin Q-functions ----
        self.pi_rob = MLPActor(
            obs_dim, act_dim, hidden_sizes, activation, self.act_limit).to(self.device)
        self.pi_adv = MLPActor(
            obs_dim, act_dim, hidden_sizes, activation, self.dist_limit).to(self.device)
        self.q1 = AdversarialMLPQFunction(
            obs_dim, act_dim, act_dim, hidden_sizes, activation).to(self.device)
        self.q2 = AdversarialMLPQFunction(
            obs_dim, act_dim, act_dim, hidden_sizes, activation).to(self.device)

        # targets
        self.pi_rob_targ = deepcopy(self.pi_rob)
        self.pi_adv_targ = deepcopy(self.pi_adv)
        self.q1_targ = deepcopy(self.q1)
        self.q2_targ = deepcopy(self.q2)
        for p in (list(self.pi_rob_targ.parameters())
                  + list(self.pi_adv_targ.parameters())
                  + list(self.q1_targ.parameters())
                  + list(self.q2_targ.parameters())):
            p.requires_grad = False

        # optional optimal policy
        self.pi_opt: Optional[MLPActor] = None
        if pi_opt_path is not None:
            self.pi_opt = MLPActor(
                obs_dim, act_dim, hidden_sizes, activation, self.act_limit).to(self.device)
            self.pi_opt.load_state_dict(
                torch.load(os.path.join(pi_opt_path, "pi.pt"),
                           map_location=self.device)
            )
            self.pi_opt.eval()
            for p in self.pi_opt.parameters():
                p.requires_grad = False
            print(f"[AdversarialTD3] Loaded pi_opt from {pi_opt_path}")

        # ---- optimizers ----
        self.pi_rob_optim = Adam(self.pi_rob.parameters(), lr=pi_lr)
        self.pi_adv_optim = Adam(self.pi_adv.parameters(), lr=pi_lr)
        self.q1_optim = Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optim = Adam(self.q2.parameters(), lr=q_lr)

        # ---- replay buffer ----
        self.buffer = ReplayBuffer(
            capacity=replay_size,
            obs_dim=obs_dim,
            total_act=act_dim * 2,
            device=self.device,
            dtype=np.float32,
            env_num=num_envs if num_envs > 1 else None,
        )

        # ---- hyperparams ----
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.n_updates = n_updates
        self.act_noise = act_noise
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.disturbance_ratio = disturbance_ratio
        self.disturbance_probability = disturbance_probability

        # TD3-specific
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self._update_count = 0

        # ---- logging ----
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._csv_path = os.path.join(log_dir, "training_returns.csv")
        self.dataset_logger = DatasetLogger(os.path.join(log_dir, "dataset"))

        # Performance logger (GPU vs CPU speed tracking)
        self.perf = PerformanceLogger(self.device, log_dir)

        # ---- transformer disturbance detector ----
        self.use_transformer = use_transformer
        self.transformer_train_interval = transformer_train_interval
        if use_transformer:
            self.detector = TransformerDisturbanceDetector(
                obs_dim=obs_dim,
                sequence_length=transformer_seq_len,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_layers,
            ).to(self.device)
            self.history_buf = HistoryBuffer(
                obs_dim, transformer_seq_len, self.device)
            self.blender = AdaptiveControllerBlender(
                self.detector, self.history_buf, smoothing_window=5
            )
            self.det_trainer = DisturbanceDetectorTrainer(
                self.detector, learning_rate=transformer_lr
            )
            self._transformer_data: list[dict] = []

        print(f"[AdversarialTD3] device={self.device}  num_envs={num_envs}  "
              f"pi_rob={count_vars(self.pi_rob)}  pi_adv={count_vars(self.pi_adv)}  "
              f"q1={count_vars(self.q1)}  q2={count_vars(self.q2)}  "
              f"policy_delay={policy_delay}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_ctrl_action(self, obs: np.ndarray, noise: float) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = self.pi_rob(obs_t).cpu().numpy()
        a += noise * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    @torch.no_grad()
    def _get_adv_action(self, obs: np.ndarray, noise: float) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = self.pi_adv(obs_t).cpu().numpy()
        a += noise * np.random.randn(*a.shape)
        return np.clip(a, -self.dist_limit, self.dist_limit)

    @torch.no_grad()
    def _get_opt_action(self, obs: np.ndarray, noise: float) -> np.ndarray:
        if self.pi_opt is None:
            return self._get_ctrl_action(obs, noise)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = self.pi_opt(obs_t).cpu().numpy()
        a += noise * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    # ------------------------------------------------------------------
    def _update(self, batch: ReplayBatch) -> Dict[str, float]:
        o = batch.obs
        o2 = batch.next_obs
        r = batch.rew.squeeze(-1)
        d = batch.done.float().squeeze(-1)
        a_ctrl = batch.act[:, : self.act_dim]
        a_dist = batch.act[:, self.act_dim:]

        # ---- critic update (both Q1 and Q2) ----
        with torch.no_grad():
            # Target policy smoothing
            a2_ctrl = self.pi_rob_targ(o2)
            eps_ctrl = torch.randn_like(a2_ctrl) * self.target_noise
            eps_ctrl = eps_ctrl.clamp(-self.noise_clip, self.noise_clip)
            a2_ctrl = (a2_ctrl + eps_ctrl).clamp(-self.act_limit,
                                                 self.act_limit)

            a2_dist = self.pi_adv_targ(o2)
            eps_dist = torch.randn_like(a2_dist) * self.target_noise
            eps_dist = eps_dist.clamp(-self.noise_clip, self.noise_clip)
            a2_dist = (a2_dist + eps_dist).clamp(-self.dist_limit,
                                                 self.dist_limit)

            # Clipped double Q-learning
            q1_next = self.q1_targ(o2, a2_ctrl, a2_dist)
            q2_next = self.q2_targ(o2, a2_ctrl, a2_dist)
            q_next = torch.min(q1_next, q2_next)
            backup = r + self.gamma * (1.0 - d) * q_next

        # Q1
        q1_val = self.q1(o, a_ctrl, a_dist)
        loss_q1 = ((q1_val - backup) ** 2).mean()

        self.q1_optim.zero_grad()
        loss_q1.backward()
        grad_q1 = _grad_norm(self.q1)
        self.q1_optim.step()

        # Q2
        q2_val = self.q2(o, a_ctrl, a_dist)
        loss_q2 = ((q2_val - backup) ** 2).mean()

        self.q2_optim.zero_grad()
        loss_q2.backward()
        grad_q2 = _grad_norm(self.q2)
        self.q2_optim.step()

        metrics = {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "q1_val": q1_val.mean().item(),
            "q2_val": q2_val.mean().item(),
            "grad_q1": grad_q1,
            "grad_q2": grad_q2,
        }

        # ---- delayed actor + adversary update (gradient flip) ----
        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            for p in self.q1.parameters():
                p.requires_grad = False

            a_rob = self.pi_rob(o)
            a_adv = self.pi_adv(o)
            q_for_actors = self.q1(o, a_rob, a_adv)

            loss_pi = -q_for_actors.mean()
            self.pi_rob_optim.zero_grad()
            self.pi_adv_optim.zero_grad()
            loss_pi.backward()

            grad_pi_rob = _grad_norm(self.pi_rob)
            grad_pi_adv = _grad_norm(self.pi_adv)

            for p in self.pi_adv.parameters():
                if p.grad is not None:
                    p.grad.mul_(-1.0)

            self.pi_rob_optim.step()
            self.pi_adv_optim.step()

            for p in self.q1.parameters():
                p.requires_grad = True

            metrics["loss_pi"] = loss_pi.item()
            metrics["grad_pi_rob"] = grad_pi_rob
            metrics["grad_pi_adv"] = grad_pi_adv

            # ---- polyak update targets (only when policy is updated) ----
            with torch.no_grad():
                for p, pt in zip(self.pi_rob.parameters(), self.pi_rob_targ.parameters()):
                    pt.data.mul_(self.polyak).add_(
                        (1.0 - self.polyak) * p.data)
                for p, pt in zip(self.pi_adv.parameters(), self.pi_adv_targ.parameters()):
                    pt.data.mul_(self.polyak).add_(
                        (1.0 - self.polyak) * p.data)
                for p, pt in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    pt.data.mul_(self.polyak).add_(
                        (1.0 - self.polyak) * p.data)
                for p, pt in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    pt.data.mul_(self.polyak).add_(
                        (1.0 - self.polyak) * p.data)

        return metrics

    # ------------------------------------------------------------------
    def _collect_detector_data(
        self, obs_seq: np.ndarray, has_dist: bool, ep_return: float
    ) -> None:
        if has_dist:
            alpha_target = np.clip(ep_return / 100.0, 0.0, 0.5)
        else:
            alpha_target = np.clip(0.5 + ep_return / 200.0, 0.5, 1.0)
        self._transformer_data.append(
            {"obs_seq": obs_seq, "dist": float(
                has_dist), "alpha": alpha_target}
        )
        if len(self._transformer_data) > 10_000:
            self._transformer_data.pop(0)

    def _train_detector(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self._transformer_data) < batch_size:
            return {}
        idxs = np.random.choice(
            len(self._transformer_data), batch_size, replace=False)
        items = [self._transformer_data[i] for i in idxs]
        obs = torch.stack([torch.as_tensor(
            x["obs_seq"], dtype=torch.float32, device=self.device) for x in items])
        dist = torch.tensor([x["dist"] for x in items],
                            dtype=torch.float32, device=self.device)
        alpha = torch.tensor([x["alpha"] for x in items],
                             dtype=torch.float32, device=self.device)
        return self.det_trainer.train_step(obs, dist, alpha)

    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None) -> None:
        epochs = epochs or self.epochs
        N = self.num_envs

        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "episode_return", "has_disturbance"])

        # Metrics CSV
        metrics_csv = os.path.join(self.log_dir, "training_metrics.csv")
        metrics_header = [
            "epoch", "step", "loss_q1", "loss_q2", "loss_pi",
            "q1_val", "q2_val",
            "grad_q1", "grad_q2", "grad_pi_rob", "grad_pi_adv",
            "mean_return", "mean_length", "episodes",
            "dist_episodes", "buffer_size", "elapsed_s",
        ]
        if self.use_transformer:
            metrics_header += ["det_loss", "det_dist_loss", "det_alpha_loss"]
        if not os.path.isfile(metrics_csv):
            with open(metrics_csv, "w", newline="") as f:
                csv.writer(f).writerow(metrics_header)

        total_transitions = self.steps_per_epoch * epochs
        total_steps = total_transitions // N

        obs, _ = self.envs.reset()  # (N, obs_dim)
        ep_rets = np.zeros(N)
        ep_lens = np.zeros(N, dtype=int)

        # Per-env disturbance flags
        has_dist = np.random.rand(N) < self.disturbance_probability

        # Per-env obs history for transformer data collection
        ep_obs_histories: list[list[np.ndarray]] = [[] for _ in range(N)]
        episode_ids = np.arange(N, dtype=int)
        next_episode_id = N

        t0 = time.time()

        # Epoch-level metric accumulators
        epoch_metrics: Dict[str, list] = defaultdict(list)
        epoch_det_metrics: Dict[str, list] = defaultdict(list)
        epoch_returns: list[float] = []
        epoch_lengths: list[int] = []
        epoch_dist_eps: int = 0
        epoch_env_steps = 0
        epoch_update_count = 0

        self.perf.start_epoch()

        for t in range(total_steps):
            global_step = t * N

            # Store obs in per-env histories
            for i in range(N):
                ep_obs_histories[i].append(obs[i].copy())

            # --- action selection ---
            if global_step < self.start_steps:
                acts_ctrl = np.stack([self.envs.single_action_space.sample()
                                      for _ in range(N)])
                acts_dist = np.zeros_like(acts_ctrl)
                for i in range(N):
                    if has_dist[i]:
                        acts_dist[i] = (self.envs.single_action_space.sample()
                                        * self.disturbance_ratio)
            else:
                acts_ctrl = self._get_ctrl_action(obs, self.act_noise)
                acts_adv = self._get_adv_action(obs, self.act_noise)
                acts_dist = np.where(
                    has_dist[:, None], acts_adv,
                    np.zeros(self.act_dim, dtype=np.float32))

            acts_total = acts_ctrl + acts_dist
            self.perf.start("env_step")
            next_obs, rews, terms, truncs, infos = self.envs.step(acts_total)
            self.perf.stop("env_step")
            epoch_env_steps += N

            # Handle auto-reset
            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                for i in range(N):
                    if infos["_final_observation"][i]:
                        real_next_obs[i] = infos["final_observation"][i]

            # Store combined action in buffer
            combined_acts = np.concatenate([acts_ctrl, acts_dist], axis=1)
            self.buffer.add_batch(
                obs, real_next_obs, combined_acts,
                rews.reshape(-1, 1),
                terms.reshape(-1, 1),
                truncs.reshape(-1, 1),
            )

            # Per-env episode tracking
            ep_rets += rews
            ep_lens += 1
            dones = terms | truncs

            for i in range(N):
                if dones[i]:
                    with open(self._csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([global_step, ep_rets[i],
                                                int(has_dist[i])])
                    epoch_returns.append(ep_rets[i])
                    epoch_lengths.append(ep_lens[i])
                    if has_dist[i]:
                        epoch_dist_eps += 1
                    self.dataset_logger.end_episode()

                    # Collect transformer data from this env's episode
                    if self.use_transformer:
                        seq_len = self.detector.sequence_length
                        if len(ep_obs_histories[i]) >= seq_len:
                            obs_seq = np.array(
                                ep_obs_histories[i][-seq_len:], dtype=np.float32)
                            self._collect_detector_data(
                                obs_seq, bool(has_dist[i]), ep_rets[i])

                    # Reset per-env state
                    ep_rets[i] = 0.0
                    ep_lens[i] = 0
                    ep_obs_histories[i] = []
                    has_dist[i] = np.random.rand(
                    ) < self.disturbance_probability
                    episode_ids[i] = next_episode_id
                    next_episode_id += 1

            obs = next_obs

            # --- network updates ---
            if global_step >= self.update_after and t % self.update_every == 0:
                for _ in range(self.n_updates):
                    self.perf.start("buffer_sample")
                    batch = self.buffer.sample(self.batch_size)
                    self.perf.stop("buffer_sample")
                    self.perf.start("update")
                    m = self._update(batch)
                    self.perf.stop("update")
                    epoch_update_count += 1
                    for k, v in m.items():
                        epoch_metrics[k].append(v)

                # train transformer periodically
                if self.use_transformer and global_step % self.transformer_train_interval < N:
                    for _ in range(50):
                        det_m = self._train_detector(batch_size=64)
                        if det_m:
                            for k, v in det_m.items():
                                epoch_det_metrics[k].append(v)

            # --- epoch boundary ---
            if (global_step + N) % self.steps_per_epoch < N:
                epoch = (global_step + N) // self.steps_per_epoch
                if epoch > 0:
                    elapsed = time.time() - t0
                    mean_ret = np.mean(
                        epoch_returns) if epoch_returns else float("nan")
                    mean_len = np.mean(
                        epoch_lengths) if epoch_lengths else float("nan")

                    avg = {k: np.mean(v)
                           for k, v in epoch_metrics.items() if v}
                    avg_det = {k: np.mean(v)
                               for k, v in epoch_det_metrics.items() if v}

                    print(f"\n[AdversarialTD3] Epoch {epoch}/{epochs}  "
                          f"elapsed={elapsed:.0f}s  "
                          f"buf={len(self.buffer)}")
                    print(f"  return   {mean_ret:>8.1f}  "
                          f"(episodes={len(epoch_returns)}  "
                          f"dist={epoch_dist_eps}  "
                          f"mean_len={mean_len:.0f})")
                    if avg:
                        print(f"  loss_q1  {avg.get('loss_q1', 0):>10.4f}   "
                              f"loss_q2  {avg.get('loss_q2', 0):>10.4f}   "
                              f"loss_pi  {avg.get('loss_pi', 0):>10.4f}")
                        print(f"  Q1_val   {avg.get('q1_val', 0):>10.2f}   "
                              f"Q2_val   {avg.get('q2_val', 0):>10.2f}")
                        print(f"  grad_q1  {avg.get('grad_q1', 0):>10.4f}   "
                              f"grad_rob {avg.get('grad_pi_rob', 0):>10.4f}   "
                              f"grad_adv {avg.get('grad_pi_adv', 0):>8.4f}")
                    if avg_det:
                        print(f"  det_loss {avg_det.get('total_loss', 0):>10.6f}   "
                              f"det_dist {avg_det.get('loss_disturbance', 0):>10.6f}   "
                              f"det_blend {avg_det.get('loss_blending', 0):>8.6f}")

                    # Log to metrics CSV
                    row = [
                        epoch, global_step,
                        f"{avg.get('loss_q1', ''):.6f}" if avg else "",
                        f"{avg.get('loss_q2', ''):.6f}" if avg else "",
                        f"{avg.get('loss_pi', ''):.6f}" if avg else "",
                        f"{avg.get('q1_val', ''):.4f}" if avg else "",
                        f"{avg.get('q2_val', ''):.4f}" if avg else "",
                        f"{avg.get('grad_q1', ''):.4f}" if avg else "",
                        f"{avg.get('grad_q2', ''):.4f}" if avg else "",
                        f"{avg.get('grad_pi_rob', ''):.4f}" if avg else "",
                        f"{avg.get('grad_pi_adv', ''):.4f}" if avg else "",
                        f"{mean_ret:.2f}" if not np.isnan(mean_ret) else "",
                        f"{mean_len:.1f}" if not np.isnan(mean_len) else "",
                        len(epoch_returns),
                        epoch_dist_eps,
                        len(self.buffer),
                        f"{elapsed:.1f}",
                    ]
                    if self.use_transformer:
                        row += [
                            f"{avg_det.get('total_loss', ''):.6f}" if avg_det else "",
                            f"{avg_det.get('loss_disturbance', ''):.6f}" if avg_det else "",
                            f"{avg_det.get('loss_blending', ''):.6f}" if avg_det else "",
                        ]
                    with open(metrics_csv, "a", newline="") as f:
                        csv.writer(f).writerow(row)

                    # Reset epoch accumulators
                    epoch_metrics = defaultdict(list)
                    epoch_det_metrics = defaultdict(list)
                    epoch_returns = []
                    epoch_lengths = []
                    epoch_dist_eps = 0

                    # Performance summary
                    self.perf.epoch_summary(
                        epoch, global_step,
                        steps_this_epoch=epoch_env_steps,
                        updates_this_epoch=epoch_update_count,
                    )
                    epoch_env_steps = 0
                    epoch_update_count = 0
                    self.perf.start_epoch()

                    if epoch % self.save_freq == 0:
                        self.save()

    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.log_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)
        torch.save(self.pi_rob.state_dict(), os.path.join(path, "pi_rob.pt"))
        torch.save(self.pi_adv.state_dict(), os.path.join(path, "pi_adv.pt"))
        torch.save(self.q1.state_dict(), os.path.join(path, "q1.pt"))
        torch.save(self.q2.state_dict(), os.path.join(path, "q2.pt"))
        if self.use_transformer:
            torch.save(self.detector.state_dict(),
                       os.path.join(path, "detector.pt"))
        print(f"  [saved] {path}")

    def load(self, path: str) -> None:
        self.pi_rob.load_state_dict(torch.load(os.path.join(
            path, "pi_rob.pt"), map_location=self.device))
        self.pi_adv.load_state_dict(torch.load(os.path.join(
            path, "pi_adv.pt"), map_location=self.device))
        self.q1.load_state_dict(torch.load(os.path.join(
            path, "q1.pt"), map_location=self.device))
        self.q2.load_state_dict(torch.load(os.path.join(
            path, "q2.pt"), map_location=self.device))
        self.pi_rob_targ = deepcopy(self.pi_rob)
        self.pi_adv_targ = deepcopy(self.pi_adv)
        self.q1_targ = deepcopy(self.q1)
        self.q2_targ = deepcopy(self.q2)
        if self.use_transformer:
            det_path = os.path.join(path, "detector.pt")
            if os.path.isfile(det_path):
                self.detector.load_state_dict(
                    torch.load(det_path, map_location=self.device))
        print(f"  [loaded] {path}")
