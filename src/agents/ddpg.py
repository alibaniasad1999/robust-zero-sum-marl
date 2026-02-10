"""
DDPG agents for the zero-sum robust RL framework.

    DDPGAgent             — trains pi_opt on the nominal environment
    AdversarialDDPGAgent  — trains pi_rob + pi_adv in a zero-sum Markov game,
                            with optional transformer-based disturbance detection
                            and policy mixing
"""

from __future__ import annotations

import csv
import os
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

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
from src.detector.transformer import (
    TransformerDisturbanceDetector,
    HistoryBuffer,
    DisturbanceDetectorTrainer,
    AdaptiveControllerBlender,
)


# ======================================================================
# DDPGAgent  — single-agent, trains pi_opt on nominal environment
# ======================================================================
class DDPGAgent:
    def __init__(
        self,
        env_fn,
        *,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
        seed: int = 0,
        steps_per_epoch: int = 4_000,
        epochs: int = 100,
        replay_size: int = 1_000_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        pi_lr: float = 1e-3,
        q_lr: float = 1e-3,
        batch_size: int = 1024,
        start_steps: int = 5_000,
        update_after: int = 10_000,
        update_every: int = 50,
        n_updates: int = 500,
        act_noise: float = 0.1,
        max_ep_len: int = 1_000,
        save_freq: int = 10,
        device: str = "auto",
        log_dir: str = "logs/nominal",
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.env = env_fn()
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.act_limit = float(self.env.action_space.high[0])

        # Networks
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, self.act_limit).to(self.device)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
        self.pi_targ = deepcopy(self.pi)
        self.q_targ = deepcopy(self.q)
        for p in list(self.pi_targ.parameters()) + list(self.q_targ.parameters()):
            p.requires_grad = False

        # Optimizers
        self.pi_optim = Adam(self.pi.parameters(), lr=pi_lr)
        self.q_optim = Adam(self.q.parameters(), lr=q_lr)

        # Replay buffer (new-style from src.utils.buffers)
        self.buffer = ReplayBuffer(
            capacity=replay_size,
            obs_dim=obs_dim,
            total_act=act_dim,
            device=self.device,
            dtype=np.float32,
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

        print(f"[DDPGAgent] device={self.device}  "
              f"pi_params={count_vars(self.pi)}  q_params={count_vars(self.q)}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_action(self, obs: np.ndarray, noise: float) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = self.pi(obs_t).cpu().numpy()
        a += noise * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    # ------------------------------------------------------------------
    def _update(self, batch: ReplayBatch) -> None:
        o, a, r = batch.obs, batch.act, batch.rew.squeeze(-1)
        o2 = batch.next_obs
        d = batch.done.float().squeeze(-1)

        # Critic
        with torch.no_grad():
            a2 = self.pi_targ(o2)
            q_targ_val = self.q_targ(o2, a2)
            backup = r + self.gamma * (1.0 - d) * q_targ_val
        q_val = self.q(o, a)
        loss_q = ((q_val - backup) ** 2).mean()

        self.q_optim.zero_grad()
        loss_q.backward()
        self.q_optim.step()

        # Actor
        for p in self.q.parameters():
            p.requires_grad = False
        loss_pi = -self.q(o, self.pi(o)).mean()
        self.pi_optim.zero_grad()
        loss_pi.backward()
        self.pi_optim.step()
        for p in self.q.parameters():
            p.requires_grad = True

        # Polyak
        with torch.no_grad():
            for p, pt in zip(self.pi.parameters(), self.pi_targ.parameters()):
                pt.data.mul_(self.polyak).add_((1.0 - self.polyak) * p.data)
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.data.mul_(self.polyak).add_((1.0 - self.polyak) * p.data)

    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None) -> None:
        epochs = epochs or self.epochs
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "episode_return"])

        total_steps = self.steps_per_epoch * epochs
        obs, _ = self.env.reset()
        ep_ret, ep_len = 0.0, 0
        t0 = time.time()

        for t in range(total_steps):
            if t < self.start_steps:
                act = self.env.action_space.sample()
            else:
                act = self._get_action(obs, self.act_noise)

            next_obs, reward, terminated, truncated, _ = self.env.step(act)
            self.buffer.add(obs, next_obs, act, reward, terminated, truncated)
            obs = next_obs
            ep_ret += reward
            ep_len += 1

            if terminated or truncated or ep_len >= self.max_ep_len:
                with open(self._csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([t, ep_ret])
                print(f"  step={t+1}  ret={ep_ret:.1f}  len={ep_len}")
                obs, _ = self.env.reset()
                ep_ret, ep_len = 0.0, 0

            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.n_updates):
                    batch = self.buffer.sample(self.batch_size)
                    self._update(batch)

            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch
                print(f"[DDPGAgent] Epoch {epoch}/{epochs}  "
                      f"elapsed={time.time()-t0:.0f}s")
                if epoch % self.save_freq == 0:
                    self.save()

    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.log_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)
        torch.save(self.pi.state_dict(), os.path.join(path, "pi.pt"))
        torch.save(self.q.state_dict(), os.path.join(path, "q.pt"))
        print(f"  [saved] {path}")

    def load(self, path: str) -> None:
        self.pi.load_state_dict(torch.load(os.path.join(path, "pi.pt"), map_location=self.device))
        self.q.load_state_dict(torch.load(os.path.join(path, "q.pt"), map_location=self.device))
        self.pi_targ = deepcopy(self.pi)
        self.q_targ = deepcopy(self.q)
        print(f"  [loaded] {path}")


# ======================================================================
# AdversarialDDPGAgent — zero-sum game: pi_rob vs pi_adv
# ======================================================================
class AdversarialDDPGAgent:
    """
    Two-player zero-sum adversarial DDPG.

    The *controller* (pi_rob) maximises cumulative reward.
    The *adversary* (pi_adv) minimises it (gradient flip).
    Optionally trains a transformer disturbance detector and uses it
    to blend pi_opt (from a separately-trained DDPGAgent) with pi_rob.
    """

    def __init__(
        self,
        env_fn,
        *,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
        seed: int = 0,
        steps_per_epoch: int = 4_000,
        epochs: int = 100,
        replay_size: int = 1_000_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        pi_lr: float = 1e-3,
        q_lr: float = 1e-3,
        batch_size: int = 1024,
        start_steps: int = 5_000,
        update_after: int = 10_000,
        update_every: int = 50,
        n_updates: int = 500,
        act_noise: float = 0.1,
        max_ep_len: int = 1_000,
        save_freq: int = 10,
        device: str = "auto",
        log_dir: str = "logs/adversarial",
        # adversary
        disturbance_ratio: float = 0.1,
        disturbance_probability: float = 0.5,
        # transformer detector
        use_transformer: bool = True,
        transformer_seq_len: int = 20,
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_layers: int = 3,
        transformer_lr: float = 1e-4,
        transformer_train_interval: int = 500,
        # pre-trained optimal policy (for blending)
        pi_opt_path: Optional[str] = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.env = env_fn()
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.act_limit = float(self.env.action_space.high[0])
        self.dist_limit = self.act_limit * disturbance_ratio

        # ---- networks ----
        # robust controller
        self.pi_rob = MLPActor(obs_dim, act_dim, hidden_sizes, activation, self.act_limit).to(self.device)
        # adversary (bounded output)
        self.pi_adv = MLPActor(obs_dim, act_dim, hidden_sizes, activation, self.dist_limit).to(self.device)
        # centralized Q(s, a_ctrl, a_dist)
        self.q = AdversarialMLPQFunction(obs_dim, act_dim, act_dim, hidden_sizes, activation).to(self.device)

        # targets
        self.pi_rob_targ = deepcopy(self.pi_rob)
        self.pi_adv_targ = deepcopy(self.pi_adv)
        self.q_targ = deepcopy(self.q)
        for p in (list(self.pi_rob_targ.parameters())
                  + list(self.pi_adv_targ.parameters())
                  + list(self.q_targ.parameters())):
            p.requires_grad = False

        # optional optimal policy (loaded separately)
        self.pi_opt: Optional[MLPActor] = None
        if pi_opt_path is not None:
            self.pi_opt = MLPActor(obs_dim, act_dim, hidden_sizes, activation, self.act_limit).to(self.device)
            self.pi_opt.load_state_dict(
                torch.load(os.path.join(pi_opt_path, "pi.pt"), map_location=self.device)
            )
            self.pi_opt.eval()
            for p in self.pi_opt.parameters():
                p.requires_grad = False
            print(f"[AdversarialDDPG] Loaded pi_opt from {pi_opt_path}")

        # ---- optimizers ----
        self.pi_rob_optim = Adam(self.pi_rob.parameters(), lr=pi_lr)
        self.pi_adv_optim = Adam(self.pi_adv.parameters(), lr=pi_lr)
        self.q_optim = Adam(self.q.parameters(), lr=q_lr)

        # ---- replay buffer ----
        # total_act = act_dim (ctrl) + act_dim (dist)
        self.buffer = ReplayBuffer(
            capacity=replay_size,
            obs_dim=obs_dim,
            total_act=act_dim * 2,
            device=self.device,
            dtype=np.float32,
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

        # ---- logging ----
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._csv_path = os.path.join(log_dir, "training_returns.csv")
        self.dataset_logger = DatasetLogger(os.path.join(log_dir, "dataset"))

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
            self.history_buf = HistoryBuffer(obs_dim, transformer_seq_len, self.device)
            self.blender = AdaptiveControllerBlender(
                self.detector, self.history_buf, smoothing_window=5
            )
            self.det_trainer = DisturbanceDetectorTrainer(
                self.detector, learning_rate=transformer_lr
            )
            self._transformer_data: list[dict] = []

        print(f"[AdversarialDDPG] device={self.device}  "
              f"pi_rob={count_vars(self.pi_rob)}  pi_adv={count_vars(self.pi_adv)}  "
              f"q={count_vars(self.q)}")

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
    def _update(self, batch: ReplayBatch) -> None:
        o = batch.obs
        o2 = batch.next_obs
        r = batch.rew.squeeze(-1)
        d = batch.done.float().squeeze(-1)
        # split stored action into ctrl and dist halves
        a_ctrl = batch.act[:, : self.act_dim]
        a_dist = batch.act[:, self.act_dim :]

        # ---- critic ----
        with torch.no_grad():
            a2_ctrl = self.pi_rob_targ(o2)
            a2_dist = self.pi_adv_targ(o2)
            q_next = self.q_targ(o2, a2_ctrl, a2_dist)
            backup = r + self.gamma * (1.0 - d) * q_next
        q_val = self.q(o, a_ctrl, a_dist)
        loss_q = ((q_val - backup) ** 2).mean()

        self.q_optim.zero_grad()
        loss_q.backward()
        self.q_optim.step()

        # ---- actor + adversary (simultaneous, gradient flip) ----
        for p in self.q.parameters():
            p.requires_grad = False

        # compute Q with current policies
        a_rob = self.pi_rob(o)
        a_adv = self.pi_adv(o)
        q_for_actors = self.q(o, a_rob, a_adv)

        # controller maximises Q  ->  loss = -Q
        loss_pi = -q_for_actors.mean()
        self.pi_rob_optim.zero_grad()
        self.pi_adv_optim.zero_grad()
        loss_pi.backward()

        # flip adversary gradients so it *minimises* Q
        for p in self.pi_adv.parameters():
            if p.grad is not None:
                p.grad.mul_(-1.0)

        self.pi_rob_optim.step()
        self.pi_adv_optim.step()

        for p in self.q.parameters():
            p.requires_grad = True

        # ---- polyak ----
        with torch.no_grad():
            for p, pt in zip(self.pi_rob.parameters(), self.pi_rob_targ.parameters()):
                pt.data.mul_(self.polyak).add_((1.0 - self.polyak) * p.data)
            for p, pt in zip(self.pi_adv.parameters(), self.pi_adv_targ.parameters()):
                pt.data.mul_(self.polyak).add_((1.0 - self.polyak) * p.data)
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.data.mul_(self.polyak).add_((1.0 - self.polyak) * p.data)

    # ------------------------------------------------------------------
    def _collect_detector_data(
        self, obs_seq: np.ndarray, has_dist: bool, ep_return: float
    ) -> None:
        if has_dist:
            alpha_target = np.clip(ep_return / 100.0, 0.0, 0.5)
        else:
            alpha_target = np.clip(0.5 + ep_return / 200.0, 0.5, 1.0)
        self._transformer_data.append(
            {"obs_seq": obs_seq, "dist": float(has_dist), "alpha": alpha_target}
        )
        if len(self._transformer_data) > 10_000:
            self._transformer_data.pop(0)

    def _train_detector(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self._transformer_data) < batch_size:
            return {}
        idxs = np.random.choice(len(self._transformer_data), batch_size, replace=False)
        items = [self._transformer_data[i] for i in idxs]
        obs = torch.stack([torch.as_tensor(x["obs_seq"], dtype=torch.float32, device=self.device) for x in items])
        dist = torch.tensor([x["dist"] for x in items], dtype=torch.float32, device=self.device)
        alpha = torch.tensor([x["alpha"] for x in items], dtype=torch.float32, device=self.device)
        return self.det_trainer.train_step(obs, dist, alpha)

    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None) -> None:
        epochs = epochs or self.epochs

        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "episode_return", "has_disturbance"])

        total_steps = self.steps_per_epoch * epochs
        obs, _ = self.env.reset()
        ep_ret, ep_len = 0.0, 0
        has_dist = np.random.rand() < self.disturbance_probability
        ep_obs_history: list[np.ndarray] = []
        episode_id = 0
        t0 = time.time()

        for t in range(total_steps):
            ep_obs_history.append(obs.copy())

            # --- action selection ---
            if t < self.start_steps:
                act_ctrl = self.env.action_space.sample()
                act_dist = (
                    self.env.action_space.sample() * self.disturbance_ratio
                    if has_dist
                    else np.zeros(self.act_dim, dtype=np.float32)
                )
            else:
                act_ctrl = self._get_ctrl_action(obs, self.act_noise)
                act_dist = (
                    self._get_adv_action(obs, self.act_noise)
                    if has_dist
                    else np.zeros(self.act_dim, dtype=np.float32)
                )

            act_total = act_ctrl + act_dist
            next_obs, reward, terminated, truncated, info = self.env.step(act_total)

            # store in replay buffer (concatenate ctrl+dist as single action vector)
            combined_act = np.concatenate([act_ctrl, act_dist])
            self.buffer.add(obs, next_obs, combined_act, reward, terminated, truncated)

            # store in dataset logger
            dist_params = np.array([float(has_dist), self.disturbance_ratio])
            self.dataset_logger.log(
                TransitionRecord(
                    t=ep_len,
                    episode_id=episode_id,
                    obs=obs,
                    action_ctrl=act_ctrl,
                    action_dist=act_dist,
                    action_total=act_total,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    disturbance_params=dist_params,
                    disturbance_tag="adversary" if has_dist else "none",
                )
            )

            obs = next_obs
            ep_ret += reward
            ep_len += 1

            # --- episode boundary ---
            if terminated or truncated or ep_len >= self.max_ep_len:
                with open(self._csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([t, ep_ret, int(has_dist)])
                print(f"  step={t+1}  ret={ep_ret:.1f}  len={ep_len}  dist={has_dist}")
                self.dataset_logger.end_episode()

                # collect transformer data
                if self.use_transformer:
                    seq_len = self.detector.sequence_length
                    if len(ep_obs_history) >= seq_len:
                        obs_seq = np.array(ep_obs_history[-seq_len:], dtype=np.float32)
                        self._collect_detector_data(obs_seq, has_dist, ep_ret)
                    self.blender.reset()

                # reset episode
                obs, _ = self.env.reset()
                ep_ret, ep_len = 0.0, 0
                ep_obs_history = []
                has_dist = np.random.rand() < self.disturbance_probability
                episode_id += 1

            # --- network updates ---
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.n_updates):
                    batch = self.buffer.sample(self.batch_size)
                    self._update(batch)

                # train transformer periodically
                if self.use_transformer and t % self.transformer_train_interval == 0:
                    for _ in range(50):
                        info_det = self._train_detector(batch_size=64)
                        if info_det:
                            pass  # suppress spam; logged in dataset

            # --- epoch boundary ---
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch
                print(f"[AdversarialDDPG] Epoch {epoch}/{epochs}  "
                      f"elapsed={time.time()-t0:.0f}s")
                if epoch % self.save_freq == 0:
                    self.save()

    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.log_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)
        torch.save(self.pi_rob.state_dict(), os.path.join(path, "pi_rob.pt"))
        torch.save(self.pi_adv.state_dict(), os.path.join(path, "pi_adv.pt"))
        torch.save(self.q.state_dict(), os.path.join(path, "q.pt"))
        if self.use_transformer:
            torch.save(self.detector.state_dict(), os.path.join(path, "detector.pt"))
        print(f"  [saved] {path}")

    def load(self, path: str) -> None:
        self.pi_rob.load_state_dict(torch.load(os.path.join(path, "pi_rob.pt"), map_location=self.device))
        self.pi_adv.load_state_dict(torch.load(os.path.join(path, "pi_adv.pt"), map_location=self.device))
        self.q.load_state_dict(torch.load(os.path.join(path, "q.pt"), map_location=self.device))
        self.pi_rob_targ = deepcopy(self.pi_rob)
        self.pi_adv_targ = deepcopy(self.pi_adv)
        self.q_targ = deepcopy(self.q)
        if self.use_transformer:
            det_path = os.path.join(path, "detector.pt")
            if os.path.isfile(det_path):
                self.detector.load_state_dict(torch.load(det_path, map_location=self.device))
        print(f"  [loaded] {path}")
