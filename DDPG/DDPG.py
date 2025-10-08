import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANSI_COLORS = dict(
    gray=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37, crimson=38
)

def colorize(text: str, color: str, bold: bool = False, highlight: bool = False) -> str:
    """Return ANSI-colored text."""
    attr = []
    num = ANSI_COLORS[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), text)


logging.getLogger("matplotlib.font_manager").setLevel(level=logging.CRITICAL)


# DDPG Core
def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=None, device=torch.device("cpu")):
    # renamed device_ -> device for consistency
    layers = []
    if output_activation is None:
        output_activation = nn.Identity
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers).to(device)


def count_vars(module: nn.Module) -> int:
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, device=torch.device("cpu")):
        # device_ -> device
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, output_activation=nn.Tanh, device=device)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device=torch.device("cpu")):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, device=device)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.Tanh, device=torch.device("cpu")):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, device).to(device)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, device).to(device)

    def act(self, obs) -> np.ndarray:
        """Return action for given observation, for exploration."""
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device=torch.device("cpu")):
        if device == torch.device("cuda"):
            self.obs_buf = torch.zeros((size, *obs_dim), dtype=torch.float32, device=device)
            self.obs2_buf = torch.zeros((size, *obs_dim), dtype=torch.float32, device=device)
            self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
            self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
            self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        else:
            self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
            self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
            self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
            self.rew_buf = np.zeros(size, dtype=np.float32)
            self.done_buf = np.zeros(size, dtype=np.float32)
            self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        if isinstance(self.obs_buf, torch.Tensor):
            self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            self.obs2_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
            self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
            self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        else:
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        idxes = np.random.randint(0, self.size, size=batch_size)
        if isinstance(self.obs_buf, torch.Tensor):
            batch = dict(
                obs=self.obs_buf[idxes],
                obs2=self.obs2_buf[idxes],
                act=self.act_buf[idxes],
                rew=self.rew_buf[idxes],
                done=self.done_buf[idxes],
            )
            return {k: v.clone().detach() for k, v in batch.items()}
        else:
            batch = dict(
                obs=self.obs_buf[idxes],
                obs2=self.obs2_buf[idxes],
                act=self.act_buf[idxes],
                rew=self.rew_buf[idxes],
                done=self.done_buf[idxes],
            )
            return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}


class DDPGAgent:
    ##################################################################################
    # DDPG Agent
    ##################################################################################
    def __init__(
        self,
        env_fn,
        actor_critic,
        ac_kwargs,
        seed=0,
        steps_per_epoch=4000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=1024,
        start_steps=5000,
        update_after=10000,
        update_every=50,
        act_noise=0.1,
        num_test_episodes=10,
        max_ep_len=1000,
        save_freq=1,
        device="auto",
        log_dir="logs",
        plot_freq=1,
    ):
        """DDPG Agent initialization."""
        # Map underscored params to clean internal names (backward compatibility).
        seed, epochs, gamma, device, ac_kwargs = seed, epochs, gamma, device, ac_kwargs
        torch.manual_seed(seed); np.random.seed(seed)

        # Device resolution
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        print(f"[DDPG] Using device: {self.device}")

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]
        self.act_limit = float(self.env.action_space.high[0])

        # Build networks on resolved device
        self.actor_critic = actor_critic(
            self.env.observation_space,
            self.env.action_space,
            **ac_kwargs,
            device=self.device
        ).to(self.device)

        self.target_actor_critic = deepcopy(self.actor_critic).to(self.device)
        for p in self.target_actor_critic.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=self.device)
        self.gamma = gamma
        self.polyak = polyak
        self.actor_optimizer = Adam(self.actor_critic.pi.parameters(), lr=pi_lr)
        self.critic_optimizer = Adam(self.actor_critic.q.parameters(), lr=q_lr)

        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        # logging additions
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_returns = []
        self.episode_steps = []
        self._csv_path = os.path.join(self.log_dir, "training_returns.csv")

        actor_params, critic_params = (count_vars(self.actor_critic.pi), count_vars(self.actor_critic.q))
        print(f"\nNumber of parameters: \t actor: {actor_params}, \t critic: {critic_params}\n")

    def _compute_critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obs, act, rew, next_obs, done = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
        q_val = self.actor_critic.q(obs, act)
        with torch.no_grad():
            target_q = self.target_actor_critic.q(next_obs, self.target_actor_critic.pi(next_obs))
            backup = rew + self.gamma * (1 - done) * target_q
        loss_q = ((q_val - backup) ** 2).mean()
        return loss_q, {"q_values": q_val.detach().cpu().numpy()}

    def _compute_actor_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs"]
        return -self.actor_critic.q(obs, self.actor_critic.pi(obs)).mean()

    def _update(self, batch: Dict[str, torch.Tensor]) -> None:
        # Critic update
        self.critic_optimizer.zero_grad()
        critic_loss, _ = self._compute_critic_loss(batch)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic, update actor
        for p in self.actor_critic.q.parameters():
            p.requires_grad = False
        self.actor_optimizer.zero_grad()
        actor_loss = self._compute_actor_loss(batch)
        actor_loss.backward()
        self.actor_optimizer.step()
        for p in self.actor_critic.q.parameters():
            p.requires_grad = True

        # Polyak averaging
        with torch.no_grad():
            for p, p_t in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
                p_t.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)

    def get_action(self, obs: np.ndarray, noise_scale: float) -> np.ndarray:
        # ensure obs tensor is created on the active device
        action = self.actor_critic.act(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        action += noise_scale * np.random.randn(self.env.action_space.shape[0])
        return np.clip(action, -self.act_limit, self.act_limit)

    def _record_episode(self, global_step: int, ep_return: float):
        self.episode_steps.append(global_step)
        self.episode_returns.append(ep_return)
        # append to csv
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([global_step, ep_return])

    def _plot_returns(self, epoch: int):
        if not self.episode_returns:
            return
        plt.figure(figsize=(6,4))
        plt.plot(self.episode_steps, self.episode_returns, label="Episode Return")
        plt.xlabel("Env Steps")
        plt.ylabel("Return")
        plt.title("Return vs Steps")
        plt.grid(True, alpha=0.3)
        plt.legend()
        latest_path = os.path.join(self.log_dir, "return_vs_step_latest.png")
        plt.savefig(latest_path, dpi=150, bbox_inches="tight")
        per_epoch_path = os.path.join(self.log_dir, f"return_vs_step_epoch_{epoch}.png")
        plt.savefig(per_epoch_path, dpi=150, bbox_inches="tight")
        plt.close()

    def train(self, epochs: int = None):
        if epochs is None:
            epochs = self.epochs
        ##################################################################################
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["global_step", "episode_return"])
        total_steps = self.steps_per_epoch * epochs
        start_time = time.time()
        obs, _ = self.env.reset()
        episode_return, episode_length = 0.0, 0

        for t in range(total_steps):
            if t > self.start_steps:
                act = self.get_action(obs, self.act_noise)
            else:
                act = self.env.action_space.sample()

            next_obs, reward, done, _, _ = self.env.step(act)
            self.replay_buffer.store(obs, act, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            episode_length += 1

            if done or (episode_length == self.max_ep_len):
                # record before reset
                self._record_episode(t, episode_return)
                print(f"Step: {t+1}, Episode Return: {episode_return:.2f}, Episode Length: {episode_length}")
                obs, _ = self.env.reset()
                episode_return, episode_length = 0.0, 0

            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(500):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self._update(batch)

            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch
                print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")


            # save model
            if (t + 1) % (self.steps_per_epoch * self.save_freq) == 0:
                self.save()

    def save(self, filepath: str = "model/"):
        os.makedirs(filepath, exist_ok=True)
        if self.device == torch.device("cuda"):
            torch.save(self.actor_critic.pi.state_dict(), os.path.join(filepath, "actor_cuda.pth"))
            torch.save(self.actor_critic.q.state_dict(), os.path.join(filepath, "q_cuda.pth"))
        else:
            torch.save(self.actor_critic.pi.state_dict(), os.path.join(filepath, "actor_cpu.pth"))
            torch.save(self.actor_critic.q.state_dict(), os.path.join(filepath, "q_cpu.pth"))
        print(colorize("Model saved.", "blue", bold=True))

    def load(self, filepath: str = "model/", load_device: torch.device = torch.device("cpu"), from_device_to_load: str = "cpu"):
        actor_file = f"actor_{from_device_to_load}.pth"
        critic_file = f"q_{from_device_to_load}.pth"
        actor_path = os.path.join(filepath, actor_file)
        critic_path = os.path.join(filepath, critic_file)

        if not (os.path.isfile(actor_path) and os.path.isfile(critic_path)):
            print(colorize("Model not found.", "red", bold=True))
            return

        map_loc = load_device
        self.actor_critic.pi.load_state_dict(torch.load(actor_path, map_location=map_loc))
        self.actor_critic.q.load_state_dict(torch.load(critic_path, map_location=map_loc))
        print(colorize(f"Model loaded on {load_device}.", "blue", bold=True))
